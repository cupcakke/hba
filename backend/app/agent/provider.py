from __future__ import annotations

import asyncio
import codecs
import json
import logging
import random
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, AsyncIterator, Awaitable, TypeVar

import httpx

from app.errors import (
    AgentFault,
    ProtocolFault,
    ProviderFault,
    RuntimeLimitFault,
)
from app.models import ModelResponse, RuntimeConfig, ToolCall

T = TypeVar("T")
logger = logging.getLogger("app.model_provider")

MAX_SSE_EVENT_BYTES = 1_048_576
MAX_OUTPUT_BYTES = 1_048_576
MAX_TOOL_NAME_CHARS = 200
MAX_TOOL_ARGUMENT_CHARS = 1_048_576
MAX_CALLS_PER_STEP = 16
MAX_UPSTREAM_RETRIES = 0
MODEL_CONNECT_TIMEOUT_SECONDS = 120
MODEL_IDLE_TIMEOUT_SECONDS = 180
MODEL_MAX_OUTPUT_TOKENS = 27_000

_shared_clients: dict[int, httpx.AsyncClient] = {}


def _get_shared_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    client = _shared_clients.get(loop_id)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=MODEL_CONNECT_TIMEOUT_SECONDS,
                read=MODEL_IDLE_TIMEOUT_SECONDS,
                write=MODEL_CONNECT_TIMEOUT_SECONDS,
                pool=MODEL_CONNECT_TIMEOUT_SECONDS,
            ),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
            follow_redirects=False,
        )
        _shared_clients[loop_id] = client
    return client


@dataclass
class ToolCallBuilder:
    id: str = ""
    name: str = ""
    name_parts: int = 0
    name_snapshot: str = ""
    name_snapshot_valid: bool = True
    name_overflow: bool = False
    arguments: str = ""
    argument_parts: int = 0
    argument_snapshot: str = ""
    argument_snapshot_valid: bool = True
    argument_overflow: bool = False


@dataclass
class ModelStream:
    response: httpx.Response
    client: httpx.AsyncClient
    touch: Any
    model: str = ""
    task_id: str = ""
    purpose: str = ""
    attempt: int = 1


async def await_with_cancel(awaitable: Awaitable[T], cancel: asyncio.Event) -> T:
    task = asyncio.ensure_future(awaitable)
    waiter = asyncio.ensure_future(cancel.wait())
    try:
        done, pending = await asyncio.wait(
            {task, waiter}, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            return task.result()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        raise AgentFault("CANCELLED", "A feladat megszakadt.", 499)
    finally:
        for pending_task in (task, waiter):
            if not pending_task.done():
                pending_task.cancel()
        if waiter.done() and not waiter.cancelled():
            try:
                waiter.exception()
            except Exception:
                pass


def _append_fragment(
    combined: str,
    snapshot: str,
    parts: int,
    snapshot_valid: bool,
    overflow: bool,
    fragment: str,
    limit: int,
) -> tuple[str, str, int, bool, bool]:
    if not fragment:
        return combined, snapshot, parts, snapshot_valid, overflow
    if len(fragment) > limit:
        raise ProtocolFault("Az eszközhívás túl nagy.")
    if parts > 0 and not fragment.startswith(snapshot):
        snapshot_valid = False
    snapshot = fragment
    parts += 1
    if not overflow:
        if len(combined) + len(fragment) > limit:
            if parts < 2 or not snapshot_valid:
                raise ProtocolFault("Az eszközhívás túl nagy.")
            overflow = True
            combined = ""
        else:
            combined += fragment
    if overflow and not snapshot_valid:
        raise ProtocolFault("Az eszközhívás túl nagy vagy következetlen.")
    return combined, snapshot, parts, snapshot_valid, overflow


def _resolve_fragment(
    combined: str,
    snapshot: str,
    parts: int,
    snapshot_valid: bool,
    overflow: bool,
    accept,
) -> str:
    snapshot_usable = parts >= 2 and snapshot_valid
    if overflow:
        return snapshot if snapshot_usable else ""
    if snapshot_usable and not accept(combined) and accept(snapshot):
        return snapshot
    return combined


def _parses_as_json(value: str) -> bool:
    try:
        json.loads(value)
        return True
    except json.JSONDecodeError:
        return False


def _is_tool_name(value: str) -> bool:
    from app.agent.tools import is_tool_name

    return is_tool_name(value)


def finalize_tool_call(builder: ToolCallBuilder) -> ToolCall:
    name = _resolve_fragment(
        builder.name,
        builder.name_snapshot,
        builder.name_parts,
        builder.name_snapshot_valid,
        builder.name_overflow,
        _is_tool_name,
    )
    arguments = _resolve_fragment(
        builder.arguments,
        builder.argument_snapshot,
        builder.argument_parts,
        builder.argument_snapshot_valid,
        builder.argument_overflow,
        _parses_as_json,
    )
    try:
        return ToolCall(
            id=builder.id,
            type="function",
            function_name=name,
            function_arguments=arguments if arguments else "{}",
        )
    except Exception as exc:
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            raise ProtocolFault(f"Érvénytelen eszközhívás: {exc}", code="MODEL_PROTOCOL") from exc
        raise


async def parse_sse(
    body: AsyncIterator[bytes], cancel: asyncio.Event, touch
) -> AsyncIterator[str]:
    buffer = ""
    data_lines: list[str] = []
    size = 0
    leading = True
    pending_cr = False
    decoder = codecs.getincrementaldecoder("utf-8")("strict")

    def parse_line(line: str) -> str | None:
        nonlocal data_lines, size
        if line == "":
            if not data_lines:
                return None
            out = "\n".join(data_lines)
            data_lines = []
            size = 0
            return out
        if line.startswith(":"):
            return None
        colon = line.find(":")
        if colon < 0:
            field = line
            value = ""
        else:
            field = line[:colon]
            value = line[colon + 1 :]
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
            size += len(value.encode("utf-8")) + 1
            if size > MAX_SSE_EVENT_BYTES:
                raise ProtocolFault("Túl nagy SSE esemény.", code="STREAM_TOO_LARGE")
        return None

    iterator = body.__aiter__()
    while True:
        try:
            chunk = await await_with_cancel(iterator.__anext__(), cancel)
        except StopAsyncIteration:
            break
        touch()
        if len(chunk) > MAX_SSE_EVENT_BYTES:
            raise ProtocolFault("Túl nagy SSE átviteli blokk.", code="STREAM_TOO_LARGE")
        try:
            buffer += decoder.decode(chunk)
        except UnicodeDecodeError as exc:
            raise ProtocolFault(
                "Érvénytelen UTF-8 a modell adatfolyamában.", code="MODEL_ENCODING"
            ) from exc
        if leading:
            leading = False
            if buffer.startswith("\ufeff"):
                buffer = buffer[1:]
        if len(buffer) > MAX_SSE_EVENT_BYTES and "\n" not in buffer and "\r" not in buffer:
            raise ProtocolFault("Túl hosszú SSE sor.", code="STREAM_TOO_LARGE")
        index = 0
        for position in range(len(buffer)):
            char = buffer[position]
            if pending_cr:
                pending_cr = False
                if char == "\n":
                    index = position + 1
                    continue
            if char != "\n" and char != "\r":
                continue
            line = buffer[index:position]
            event = parse_line(line)
            if event is not None:
                yield event
            pending_cr = char == "\r"
            index = position + 1
        buffer = buffer[index:]
    try:
        buffer += decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise ProtocolFault(
            "Érvénytelen záró UTF-8 szekvencia.", code="MODEL_ENCODING"
        ) from exc
    if buffer:
        event = parse_line(buffer)
        if event is not None:
            yield event
    trailing = parse_line("")
    if trailing is not None:
        yield trailing


def retry_delay(response: httpx.Response | None, attempt: int) -> float:
    wait = -1.0
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            header = header.strip()
            if header.isdigit():
                wait = float(header) * 1000
            else:
                try:
                    parsed = parsedate_to_datetime(header)
                    if parsed is not None:
                        import time as _time

                        wait = max(0.0, parsed.timestamp() * 1000 - _time.time() * 1000)
                except (TypeError, ValueError):
                    wait = -1.0
    if wait < 0:
        wait = min(30_000, 1000 * 2**attempt) * (0.5 + random.random() * 0.5)
    return min(60_000, wait) / 1000


def _now() -> float:
    return asyncio.get_running_loop().time()


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ProtocolFault("Érvénytelen modellüzenet.")
        role = message.get("role")
        content = message.get("content")
        if content is None:
            content = ""
        if role == "tool":
            if not isinstance(content, str):
                raise ProtocolFault("Az eszközeredmény tartalma nem szöveg.")
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ProtocolFault("Az eszközeredményből hiányzik a hívásazonosító.")
            output.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content,
                }
            )
            continue
        if role not in ("system", "developer", "user", "assistant"):
            raise ProtocolFault("Nem támogatott modellüzenet-szerep.")
        normalized_content = _responses_message_content(role, content)
        if normalized_content or not message.get("tool_calls"):
            output.append({"role": role, "content": normalized_content})
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise ProtocolFault("Érvénytelen eszközhívás-lista.")
        for call in tool_calls:
            if not isinstance(call, dict):
                raise ProtocolFault("Érvénytelen eszközhívás.")
            function = call.get("function")
            if not isinstance(function, dict):
                raise ProtocolFault("Érvénytelen eszközfüggvény.")
            call_id = call.get("id")
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(call_id, str) or not call_id:
                raise ProtocolFault("Az eszközhívásból hiányzik az azonosító.")
            if not isinstance(name, str) or not name:
                raise ProtocolFault("Az eszközhívásból hiányzik a név.")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            if not isinstance(arguments, str):
                raise ProtocolFault("Az eszközhívás argumentuma nem szöveg.")
            if len(arguments) > MAX_TOOL_ARGUMENT_CHARS:
                raise ProtocolFault("Az eszközhívás argumentuma túl nagy.")
            output.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
            )
    return output


def _responses_message_content(role: str, content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ProtocolFault("A modellüzenet tartalma hibás.")
    if role != "user":
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                raise ProtocolFault("A modellüzenet tartalomeleme hibás.")
            part_type = part.get("type")
            if part_type in ("text", "input_text", "output_text", "text_file"):
                text = part.get("text")
                if not isinstance(text, str):
                    raise ProtocolFault("A modellüzenet szövegrésze hibás.")
                text_parts.append(text)
            elif part_type in ("image_url", "input_image") or part.get("image_url"):
                image = part.get("image_url")
                if isinstance(image, dict):
                    label = image.get("name") or part.get("name")
                else:
                    label = part.get("name")
                text_parts.append(
                    f"[A korábbi asszisztensüzenet képet tartalmazott: {label}]"
                    if isinstance(label, str) and label
                    else "[A korábbi asszisztensüzenet képet tartalmazott.]"
                )
            else:
                raise ProtocolFault("A modellüzenet nem támogatott tartalomtípust tartalmaz.")
        return "\n".join(text_parts)
    response_parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            raise ProtocolFault("A modellüzenet tartalomeleme hibás.")
        part_type = part.get("type")
        if part_type in ("text", "input_text", "output_text", "text_file"):
            text = part.get("text")
            if not isinstance(text, str):
                raise ProtocolFault("A modellüzenet szövegrésze hibás.")
            response_parts.append({"type": "input_text", "text": text})
            continue
        if part_type not in ("image_url", "input_image") and not part.get("image_url"):
            raise ProtocolFault("A modellüzenet nem támogatott tartalomtípust tartalmaz.")
        image = part.get("image_url")
        if isinstance(image, dict):
            image_url = image.get("url") or image.get("image_url")
            detail = image.get("detail", part.get("detail"))
        elif isinstance(image, str):
            image_url = image
            detail = part.get("detail")
        else:
            image_url = part.get("url")
            detail = part.get("detail")
        if not isinstance(image_url, str) or not image_url:
            raise ProtocolFault("A csatolt kép címe hiányzik.")
        image_part: dict[str, Any] = {
            "type": "input_image",
            "image_url": image_url,
        }
        if detail in ("auto", "low", "high"):
            image_part["detail"] = detail
        response_parts.append(image_part)
    return response_parts


def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    response_tools: list[dict[str, Any]] = [
        {"type": "web_search", "search_context_size": "high"}
    ]
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ProtocolFault("Nem támogatott modell-eszköztípus.")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ProtocolFault("Érvénytelen modell-eszköz.")
        name = function.get("name")
        description = function.get("description")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name:
            raise ProtocolFault("Érvénytelen modell-eszköznév.")
        if not isinstance(description, str):
            raise ProtocolFault("Érvénytelen modell-eszközleírás.")
        if not isinstance(parameters, dict):
            raise ProtocolFault("Érvénytelen modell-eszközparaméter.")
        response_tools.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )
    return response_tools


def _response_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    prompt_tokens = value.get("input_tokens")
    completion_tokens = value.get("output_tokens")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 0
        or isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens < 0
    ):
        return None
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    input_details = value.get("input_tokens_details")
    if isinstance(input_details, dict):
        cached_tokens = input_details.get("cached_tokens")
        if isinstance(cached_tokens, int) and not isinstance(cached_tokens, bool):
            usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    output_details = value.get("output_tokens_details")
    if isinstance(output_details, dict):
        reasoning_tokens = output_details.get("reasoning_tokens")
        if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
            usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return usage


def _select_reasoning_effort(purpose: str, escalated: bool = False) -> str:
    from app.config import get_settings

    settings = get_settings()
    if escalated:
        return settings.model_reasoning_effort_escalated
    if purpose != "agent_step":
        return settings.model_reasoning_effort_cognitive
    return settings.model_reasoning_effort


async def call_model(
    config: RuntimeConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    cancel: asyncio.Event,
    deadline: float,
    *,
    task_id: str = "",
    purpose: str = "agent_step",
    escalated: bool = False,
) -> ModelStream:
    from app.config import get_settings

    settings = get_settings()
    effort = _select_reasoning_effort(purpose, escalated)
    summary_mode = settings.model_reasoning_summary or "detailed"
    payload = {
        "model": settings.model_name,
        "input": _responses_input(messages),
        "stream": True,
        "temperature": 1,
        "max_output_tokens": MODEL_MAX_OUTPUT_TOKENS,
        "top_p": 1,
        "reasoning": {"effort": effort, "summary": summary_mode},
        "tools": _responses_tools(tools),
    }
    logger.info(
        "model_reasoning_effort_selected",
        extra={"job_id": task_id, "purpose": purpose, "effort": effort, "escalated": escalated},
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    client = _get_shared_client()
    for attempt in range(MAX_UPSTREAM_RETRIES + 1):
        attempt_started = time.monotonic()
        if cancel.is_set():
            raise AgentFault("CANCELLED", "A feladat megszakadt.", 499)
        if deadline - _now() <= 0:
            raise RuntimeLimitFault()
        request = client.build_request(
            "POST",
            settings.model_api_url,
            headers={
                "Authorization": f"Bearer {config.model_key.get_secret_value()}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            content=body,
        )
        logger.info(
            "model_request_started",
            extra={
                "job_id": task_id,
                "model": settings.model_name,
                "purpose": purpose,
                "attempt": attempt + 1,
                "prompt_chars": sum(
                    len(str(message.get("content") or "")) for message in messages
                ),
                "tool_count": len(tools),
            },
        )
        try:
            response = await client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            logger.warning(
                "model_request_failed",
                extra={
                    "job_id": task_id,
                    "model": settings.model_name,
                    "purpose": purpose,
                    "attempt": attempt + 1,
                    "error_code": "MODEL_TIMEOUT",
                    "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                },
            )
            if attempt < MAX_UPSTREAM_RETRIES:
                await asyncio.sleep(retry_delay(None, attempt))
                continue
            raise ProviderFault("MODEL_TIMEOUT", "A modell nem válaszolt időben.", 504) from exc
        except httpx.HTTPError as exc:
            if cancel.is_set():
                raise AgentFault("CANCELLED", "A feladat megszakadt.", 499) from exc
            logger.warning(
                "model_request_failed",
                extra={
                    "job_id": task_id,
                    "model": settings.model_name,
                    "purpose": purpose,
                    "attempt": attempt + 1,
                    "error_code": "MODEL_TRANSPORT",
                    "error_type": type(exc).__name__,
                    "duration_ms": int((time.monotonic() - attempt_started) * 1000),
                },
            )
            if attempt < MAX_UPSTREAM_RETRIES:
                await asyncio.sleep(retry_delay(None, attempt))
                continue
            raise ProviderFault("MODEL_TRANSPORT", "A modellhez nem lehet csatlakozni.") from exc
        logger.info(
            "model_request_http",
            extra={
                "job_id": task_id,
                "model": settings.model_name,
                "purpose": purpose,
                "attempt": attempt + 1,
                "status_code": response.status_code,
                "duration_ms": int((time.monotonic() - attempt_started) * 1000),
            },
        )
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" not in content_type.lower():
                await response.aclose()
                raise ProtocolFault("A modell nem SSE adatfolyamot adott vissza.")
            return ModelStream(
                response=response,
                client=client,
                touch=lambda: None,
                model=settings.model_name,
                task_id=task_id,
                purpose=purpose,
                attempt=attempt + 1,
            )
        status = response.status_code
        retryable = status in (429, 502, 503, 504)
        try:
            await response.aread()
        finally:
            await response.aclose()
        if status in (401, 403):
            raise ProviderFault("MODEL_AUTH", "Érvénytelen modell jogosultság.")
        if status == 402:
            raise ProviderFault(
                "MODEL_PAYMENT_REQUIRED",
                "A fő modell API HTTP 402 (Payment Required) választ adott. Ellenőrizd a "
                "modellszolgáltatói fiók számlázását, egyenlegét és a modellhez való hozzáférést.",
                502,
            )
        if not retryable or attempt >= MAX_UPSTREAM_RETRIES:
            if status == 429:
                raise ProviderFault("MODEL_RATE_LIMIT", f"A modell HTTP {status} hibát adott.")
            raise ProviderFault("MODEL_HTTP", f"A modell HTTP {status} hibát adott.")
        await asyncio.sleep(retry_delay(response, attempt))
    raise ProviderFault("MODEL_RETRIES_EXHAUSTED", "Elfogyott a modell újrapróbálkozási kerete.")


async def receive_model(
    stream: ModelStream,
    task_id: str,
    step: int,
    send,
    cancel: asyncio.Event,
) -> ModelResponse:
    completed = False
    content = ""
    reasoning_text = ""
    reasoning_summary_text = ""
    content_bytes = 0
    usage: dict[str, Any] | None = None
    tool_items: dict[str, ToolCallBuilder] = {}
    # Track emitted offsets per (output_index, summary_index) for summary deltas
    emitted_summary_offsets: dict[tuple[int, int], int] = {}
    # Track summary parts for joining
    summary_parts: list[str] = []

    def store_function_call(item: Any, final: bool = False) -> None:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return
        item_id = item.get("id")
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            call_id = item_id
        if not isinstance(call_id, str) or not call_id:
            raise ProtocolFault("Hiányzik a modell eszközhívás-azonosítója.")
        if not isinstance(item_id, str) or not item_id:
            item_id = call_id
        builder = tool_items.setdefault(item_id, ToolCallBuilder())
        if len(tool_items) > MAX_CALLS_PER_STEP:
            raise ProtocolFault("Túl sok modell eszközhívás.")
        if builder.id and builder.id != call_id:
            raise ProtocolFault("Ütköző modell eszközhívás-azonosító.")
        builder.id = call_id
        name = item.get("name")
        if name is not None:
            if not isinstance(name, str) or not name:
                raise ProtocolFault("Érvénytelen modell eszközhívás-név.")
            builder.name = name
        arguments = item.get("arguments")
        if arguments is not None:
            if not isinstance(arguments, str):
                raise ProtocolFault("Érvénytelen modell eszközhívás-argumentum.")
            if len(arguments) > MAX_TOOL_ARGUMENT_CHARS:
                raise ProtocolFault("Az eszközhívás argumentuma túl nagy.")
            if final or not builder.arguments:
                builder.arguments = arguments
                builder.argument_snapshot = arguments
                builder.argument_parts = 1 if arguments else 0
                builder.argument_snapshot_valid = True
                builder.argument_overflow = False

    def output_text(response: dict[str, Any]) -> str:
        pieces: list[str] = []
        output_items = response.get("output")
        if not isinstance(output_items, list):
            return ""
        for item in output_items:
            if not isinstance(item, dict):
                continue
            store_function_call(item, final=True)
            if item.get("type") != "message":
                continue
            parts = item.get("content")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if part.get("type") in ("output_text", "text") and isinstance(text, str):
                    pieces.append(text)
        return "".join(pieces)

    def output_reasoning_summary(response: dict[str, Any]) -> str:
        pieces: list[str] = []
        output_items = response.get("output")
        if not isinstance(output_items, list):
            return ""
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                continue
            summaries = item.get("summary")
            if not isinstance(summaries, list):
                continue
            for summary_item in summaries:
                if not isinstance(summary_item, dict):
                    continue
                text = summary_item.get("text")
                if summary_item.get("type") == "summary_text" and isinstance(text, str):
                    pieces.append(text)
        return "".join(pieces)

    def output_reasoning_text(response: dict[str, Any]) -> str:
        pieces: list[str] = []
        output_items = response.get("output")
        if not isinstance(output_items, list):
            return ""
        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                continue
            parts = item.get("content")
            if isinstance(parts, str):
                pieces.append(parts)
                continue
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if part.get("type") in ("reasoning_text", "text") and isinstance(text, str):
                    pieces.append(text)
        return "".join(pieces)

    async def emit_reasoning(text: str, reasoning_type: str) -> None:
        if text:
            await send(
                {
                    "type": "delta",
                    "task_id": task_id,
                    "step": step,
                    "reasoning_content": text,
                    "reasoning_type": reasoning_type,
                }
            )

    async def emit_draft(text: str) -> None:
        if text:
            await send(
                {
                    "type": "delta",
                    "task_id": task_id,
                    "step": step,
                    "content": text,
                    "draft": True,
                }
            )

    try:
        async for data in parse_sse(stream.response.aiter_bytes(), cancel, stream.touch):
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ProtocolFault(
                    "Hibás JSON az SSE eseményben.", code="MODEL_PROTOCOL"
                ) from exc
            if not isinstance(event, dict):
                raise ProtocolFault("Érvénytelen SSE esemény.", code="MODEL_PROTOCOL")
            event_type = event.get("type")
            if not isinstance(event_type, str):
                raise ProtocolFault("Hiányzik a Meta Responses eseménytípusa.")
            if event_type in ("error", "response.failed"):
                error_obj = event.get("error") if isinstance(event.get("error"), dict) else {}
                code = error_obj.get("code") if isinstance(error_obj, dict) else None
                msg = error_obj.get("message") if isinstance(error_obj, dict) else None
                logger.warning(
                    "model_error_event",
                    extra={
                        "job_id": task_id,
                        "error_code": code,
                        "error_message": (str(msg)[:200] if msg else ""),
                    },
                )
                raise ProviderFault("MODEL_ERROR", f"A fő modell hibát jelzett: {msg or code or 'ismeretlen'}", 502)
            if event_type == "response.incomplete":
                response = event.get("response")
                details = response.get("incomplete_details") if isinstance(response, dict) else None
                reason = details.get("reason") if isinstance(details, dict) else None
                if reason in ("max_output_tokens", "max_tokens"):
                    raise ProviderFault(
                        "MODEL_TRUNCATED",
                        "A modell válasza elérte a kimeneti tokenkorlátot.",
                        422,
                    )
                raise ProviderFault("MODEL_INCOMPLETE", "A modell válasza nem teljes.", 422)
            if event_type == "response.output_text.delta":
                delta = event.get("delta")
                if not isinstance(delta, str):
                    raise ProtocolFault("Érvénytelen modell-szövegfragmentum.")
                content += delta
                content_bytes += len(delta.encode("utf-8"))
                if content_bytes > MAX_OUTPUT_BYTES:
                    raise ProviderFault("MODEL_OUTPUT_LIMIT", "A modell kimenete túl hosszú.")
                if delta:
                    await emit_draft(delta)
                continue
            if event_type == "response.reasoning_summary_text.delta":
                delta = event.get("delta")
                if not isinstance(delta, str):
                    raise ProtocolFault("Érvénytelen modell-összefoglaló.")
                if delta:
                    # track offsets per indices if present
                    out_idx = event.get("output_index", 0)
                    sum_idx = event.get("summary_index", 0)
                    try:
                        out_idx = int(out_idx)
                        sum_idx = int(sum_idx)
                    except Exception:
                        out_idx, sum_idx = 0, 0
                    key = (out_idx, sum_idx)
                    prev = emitted_summary_offsets.get(key, 0)
                    # emit delta immediately
                    await emit_reasoning(delta, "summary")
                    reasoning_summary_text += delta
                    emitted_summary_offsets[key] = prev + len(delta)
                continue
            if event_type in ("response.reasoning_summary_part.added", "response.reasoning_summary_part.done"):
                part = event.get("part") if isinstance(event.get("part"), dict) else event
                text = part.get("text") if isinstance(part, dict) else None
                if isinstance(text, str) and text:
                    # emit immediately with newline handling between parts
                    if summary_parts:
                        text = "\n\n" + text
                    summary_parts.append(text)
                    reasoning_summary_text += text
                    await emit_reasoning(text, "summary")
                continue
            if event_type == "response.reasoning_summary_text.done":
                final_summary = event.get("text")
                if not isinstance(final_summary, str):
                    final_summary = event.get("summary")
                if isinstance(final_summary, str) and final_summary:
                    # emit only suffix not yet emitted
                    if final_summary.startswith(reasoning_summary_text):
                        delta = final_summary[len(reasoning_summary_text) :]
                    elif not reasoning_summary_text:
                        delta = final_summary
                    else:
                        delta = ""
                    if delta:
                        await emit_reasoning(delta, "summary")
                        reasoning_summary_text += delta
                continue
            if event_type in (
                "response.reasoning_text.delta",
                "response.reasoning.delta",
            ):
                delta = event.get("delta")
                if isinstance(delta, dict):
                    delta = delta.get("text") or delta.get("content")
                if not isinstance(delta, str):
                    raise ProtocolFault("Érvénytelen modell-gondolatmenet fragmentum.")
                if delta:
                    reasoning_text += delta
                    await emit_reasoning(delta, "raw")
                continue
            if event_type in (
                "response.reasoning_text.done",
                "response.reasoning.done",
            ):
                final_text = event.get("text")
                if not isinstance(final_text, str):
                    final_text = event.get("reasoning")
                if isinstance(final_text, str) and final_text:
                    delta = (
                        final_text[len(reasoning_text) :]
                        if final_text.startswith(reasoning_text)
                        else final_text if not reasoning_text else ""
                    )
                    if delta:
                        await emit_reasoning(delta, "raw")
                        reasoning_text += delta
                continue
            if event_type in ("response.web_search_call.in_progress", "response.web_search_call.searching"):
                await send(
                    {
                        "type": "agent_event",
                        "task_id": task_id,
                        "step": step,
                        "event": {"type": "tool_start", "tool": "web_search"},
                    }
                )
                continue
            if event_type == "response.web_search_call.completed":
                await send(
                    {
                        "type": "agent_event",
                        "task_id": task_id,
                        "step": step,
                        "event": {"type": "tool_result", "tool": "web_search", "output": event.get("output") or ""},
                    }
                )
                continue
            if event_type == "response.output_item.added":
                store_function_call(event.get("item"))
                continue
            if event_type in (
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            ):
                item_id = event.get("item_id")
                if not isinstance(item_id, str) or not item_id:
                    raise ProtocolFault("A modell eszközargumentumából hiányzik az elemazonosító.")
                builder = tool_items.setdefault(item_id, ToolCallBuilder())
                if len(tool_items) > MAX_CALLS_PER_STEP:
                    raise ProtocolFault("Túl sok modell eszközhívás.")
                arguments = event.get("delta") if event_type.endswith(".delta") else event.get("arguments")
                if not isinstance(arguments, str):
                    raise ProtocolFault("Érvénytelen modell-eszközargumentum.")
                if len(arguments) > MAX_TOOL_ARGUMENT_CHARS:
                    raise ProtocolFault("Az eszközhívás argumentuma túl nagy.")
                if event_type.endswith(".delta"):
                    (
                        builder.arguments,
                        builder.argument_snapshot,
                        builder.argument_parts,
                        builder.argument_snapshot_valid,
                        builder.argument_overflow,
                    ) = _append_fragment(
                        builder.arguments,
                        builder.argument_snapshot,
                        builder.argument_parts,
                        builder.argument_snapshot_valid,
                        builder.argument_overflow,
                        arguments,
                        MAX_TOOL_ARGUMENT_CHARS,
                    )
                else:
                    builder.arguments = arguments
                    builder.argument_snapshot = arguments
                    builder.argument_parts = 1 if arguments else 0
                    builder.argument_snapshot_valid = True
                    builder.argument_overflow = False
                continue
            if event_type == "response.output_item.done":
                store_function_call(event.get("item"), final=True)
                continue
            if event_type == "response.completed":
                response = event.get("response")
                if not isinstance(response, dict) or response.get("status") not in (
                    None,
                    "completed",
                ):
                    raise ProtocolFault("A modell válasza érvénytelen lezárást adott.")
                final_summary = output_reasoning_summary(response)
                final_reasoning = output_reasoning_text(response)
                if final_reasoning:
                    reasoning_delta = (
                        final_reasoning[len(reasoning_text) :]
                        if final_reasoning.startswith(reasoning_text)
                        else final_reasoning if not reasoning_text else ""
                    )
                    if reasoning_delta:
                        await emit_reasoning(reasoning_delta, "raw")
                        reasoning_text += reasoning_delta
                # For summary, emit only suffix not yet emitted
                if final_summary and final_summary != reasoning_summary_text:
                    if final_summary.startswith(reasoning_summary_text):
                        delta = final_summary[len(reasoning_summary_text) :]
                    elif not reasoning_summary_text:
                        delta = final_summary
                    else:
                        # suffix handling with parts joining
                        delta = ""
                        # if we have summary parts, try to detect missing
                        if final_summary not in reasoning_summary_text:
                            delta = final_summary[len(reasoning_summary_text):] if len(final_summary) > len(reasoning_summary_text) else ""
                    if delta:
                        await emit_reasoning(delta, "summary")
                        reasoning_summary_text += delta
                final_content = output_text(response)
                if final_content and final_content != content:
                    if final_content.startswith(content):
                        delta = final_content[len(content) :]
                        content += delta
                        content_bytes += len(delta.encode("utf-8"))
                        if content_bytes > MAX_OUTPUT_BYTES:
                            raise ProviderFault("MODEL_OUTPUT_LIMIT", "A modell kimenete túl hosszú.")
                        if delta:
                            await emit_draft(delta)
                    else:
                        # reconcile: log and keep streamed text
                        logger.warning(
                            "model_final_text_mismatch",
                            extra={"job_id": task_id, "streamed": len(content), "final": len(final_content)},
                        )
                usage = _response_usage(response.get("usage"))
                completed = True
                break
            if event_type == "response.cancelled":
                raise AgentFault("CANCELLED", "A feladat megszakadt.", 499)
            if event_type in (
                "response.created",
                "response.in_progress",
                "response.content_part.added",
                "response.content_part.done",
                "response.output_text.done",
            ):
                continue
    except AgentFault as fault:
        logger.warning(
            "model_response_failed",
            extra={
                "job_id": task_id,
                "model": stream.model,
                "purpose": stream.purpose,
                "step": step,
                "attempt": stream.attempt,
                "error_code": fault.code,
            },
        )
        raise
    finally:
        try:
            await stream.response.aclose()
        except Exception:
            pass
        # Do not close shared client; keep alive. Only close if it's a temporary client
        # For shared client we keep it open. Detect if client is shared.
        loop_id = id(asyncio.get_running_loop()) if asyncio.get_running_loop() else None
        if loop_id is None or _shared_clients.get(loop_id) is not stream.client:
            try:
                await stream.client.aclose()
            except Exception:
                pass
    if not completed:
        raise ProtocolFault(
            "A modell adatfolyama nem fejeződött be szabályosan.", code="MODEL_INCOMPLETE"
        )
    try:
        calls = [finalize_tool_call(builder) for builder in tool_items.values()]
    except Exception as exc:
        raise ProtocolFault(f"Érvénytelen eszközhívás: {exc}", code="MODEL_PROTOCOL") from exc
    if len(calls) > MAX_CALLS_PER_STEP:
        raise ProtocolFault("Túl sok modell eszközhívás.")
    if any(not call.id or not call.function_name for call in calls):
        raise ProtocolFault("Hiányos eszközhívási adatok.", code="MODEL_PROTOCOL")
    if len({call.id for call in calls}) != len(calls):
        raise ProtocolFault("Ismételt eszközazonosító.", code="MODEL_PROTOCOL")
    logger.info(
        "model_response_usage",
        extra={
            "job_id": task_id,
            "model": stream.model,
            "purpose": stream.purpose,
            "step": step,
            "attempt": stream.attempt,
            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
            "completion_tokens": usage.get("completion_tokens") if usage else None,
            "reasoning_tokens": (
                usage.get("completion_tokens_details", {}).get("reasoning_tokens")
                if usage
                else None
            ),
            "total_tokens": usage.get("total_tokens") if usage else None,
        },
    )
    finish = "tool_calls" if calls else "stop"
    try:
        return ModelResponse(content=content, calls=calls, finish=finish, usage=usage)
    except Exception as exc:
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            raise ProtocolFault(f"Érvénytelen modellválasz: {exc}", code="MODEL_PROTOCOL") from exc
        raise
