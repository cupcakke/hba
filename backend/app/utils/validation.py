from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Any, NoReturn, TypeGuard
from urllib.parse import urlsplit

from fastapi import Request

from app.errors import PayloadTooLargeFault, RequestFault
from app.utils.security import LOOSE_UUID_PATTERN

MAX_TEXT_LENGTH = 400_000
MAX_TOTAL_TEXT_LENGTH = 4_000_000
MAX_HISTORY_MESSAGES = 300
MAX_BODY_BYTES = 32 * 1024 * 1024
ALLOWED_CLIENT_ROLES = ("user", "assistant")
MAX_CONTENT_LENGTH_DIGITS = 12
MAX_CONTENT_PARTS = 100
IMAGE_DATA_URL_PATTERN = re.compile(
    r"^data:image/(?:png|jpe?g|gif|webp|avif);base64,([A-Za-z0-9+/]*={0,2})$",
    re.IGNORECASE,
)


def is_record(value: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict)


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"Nem megengedett JSON konstans: {name}")


def _decode_utf8(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RequestFault("Hibás UTF-8 kérés.") from exc


def _loads(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise RequestFault("Hibás JSON kérés.") from exc


def parse_json_body(raw: bytes, limit: int = MAX_BODY_BYTES) -> Any:
    if limit <= 0:
        raise ValueError("A kérésméret korlátjának pozitívnak kell lennie.")
    if not raw:
        raise RequestFault("Hiányzó kéréstartalom.")
    if len(raw) > limit:
        raise PayloadTooLargeFault()
    return _loads(_decode_utf8(raw))


def _declared_length(raw: Any) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise RequestFault("Hibás Content-Length fejléc.", code="INVALID_CONTENT_LENGTH")
    text = raw.strip()
    if not text.isascii() or not text.isdigit() or len(text) > MAX_CONTENT_LENGTH_DIGITS:
        raise RequestFault("Hibás Content-Length fejléc.", code="INVALID_CONTENT_LENGTH")
    return int(text)


async def read_request_body(request: Request, limit: int) -> bytes:
    if limit <= 0:
        raise ValueError("A kérésméret korlátjának pozitívnak kell lennie.")
    declared = _declared_length(request.headers.get("content-length"))
    if declared is not None and declared > limit:
        raise PayloadTooLargeFault()
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeFault()
        chunks.append(chunk)
    return b"".join(chunks)


def _encode_check(content: str) -> None:
    try:
        content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RequestFault("Az üzenet tartalma nem kódolható UTF-8 szövegként.") from exc


def content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for part in value:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def content_text_length(value: Any) -> int:
    return len(content_text(value))


def _validate_image_url(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_BODY_BYTES:
        raise RequestFault("A csatolt kép címe hibás vagy túl nagy.")
    if value.startswith("image-storage://"):
        raise RequestFault("A csatolt kép nem érhető el. Töltsd fel újra.")
    data_match = IMAGE_DATA_URL_PATTERN.fullmatch(value)
    if data_match is not None:
        encoded = data_match.group(1)
        if not encoded or len(encoded) % 4 == 1:
            raise RequestFault("A csatolt kép adatai hibásak.")
        try:
            base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RequestFault("A csatolt kép adatai hibásak.") from exc
        return value
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise RequestFault("A csatolt kép címe hibás.") from exc
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None and not 0 < port <= 65535
    ):
        raise RequestFault("A csatolt képnek elérhető HTTP(S) címre kell mutatnia.")
    return value


def validate_client_content(value: Any) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        if len(value) > MAX_TEXT_LENGTH or not value.strip():
            raise RequestFault("Az üzenet tartalma hibás vagy túl nagy.")
        _encode_check(value)
        return value
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_CONTENT_PARTS:
        raise RequestFault("Az üzenet tartalma hibás vagy túl nagy.")
    validated: list[dict[str, Any]] = []
    text_length = 0
    has_image = False
    for part in value:
        if not is_record(part):
            raise RequestFault("Az üzenettartalom eleme hibás.")
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str) or len(text) > MAX_TEXT_LENGTH:
                raise RequestFault("Az üzenet szövegrésze hibás vagy túl nagy.")
            _encode_check(text)
            text_length += len(text)
            validated.append({"type": "text", "text": text})
            continue
        if part_type == "text_file":
            name = part.get("name", "clipboard.txt")
            text = part.get("text")
            if (
                not isinstance(name, str)
                or len(name) > 200
                or not isinstance(text, str)
                or len(text) > MAX_TEXT_LENGTH
            ):
                raise RequestFault("A szövegfájl hibás vagy túl nagy.")
            _encode_check(name)
            _encode_check(text)
            rendered = f"Attached text file: {name}\n\n{text}"
            text_length += len(rendered)
            validated.append({"type": "text", "text": rendered})
            continue
        if part_type not in ("image_url", "input_image"):
            raise RequestFault("Az üzenet nem támogatott tartalomtípust tartalmaz.")
        image = part.get("image_url")
        detail = part.get("detail")
        if isinstance(image, dict):
            url = image.get("url") or image.get("image_url")
            detail = image.get("detail", detail)
        elif isinstance(image, str):
            url = image
        else:
            url = part.get("url")
        if detail is not None and detail not in ("auto", "low", "high"):
            raise RequestFault("A csatolt kép részletességi beállítása hibás.")
        validated_image: dict[str, Any] = {
            "url": _validate_image_url(url),
        }
        if detail is not None:
            validated_image["detail"] = detail
        validated.append({"type": "image_url", "image_url": validated_image})
        has_image = True
    if text_length > MAX_TOTAL_TEXT_LENGTH or (text_length == 0 and not has_image):
        raise RequestFault("Az üzenet tartalma hibás vagy túl nagy.")
    return validated


def validate_client_history(payload: Any) -> list[dict[str, Any]]:
    if not is_record(payload):
        raise RequestFault("Hiányzó vagy hibás üzenetlista.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= MAX_HISTORY_MESSAGES:
        raise RequestFault("Hiányzó vagy hibás üzenetlista.")
    validated: list[dict[str, Any]] = []
    last_role: str | None = None
    total_length = 0
    for raw in messages:
        if not is_record(raw):
            raise RequestFault("Az üzenet csak objektum lehet.")
        for key in raw:
            if key not in ("role", "content"):
                raise RequestFault("Kliens oldali utasítás- vagy eszköztörténet nem fogadható el.")
        role = raw.get("role")
        if not isinstance(role, str) or role not in ALLOWED_CLIENT_ROLES:
            raise RequestFault("Az üzenet szerepköre nem megengedett.")
        if last_role is None and role != "user":
            raise RequestFault("Az első üzenetnek felhasználói üzenetnek kell lennie.")
        if last_role is not None and role == last_role:
            raise RequestFault("Az üzenetek szerepkörének váltakoznia kell.")
        content = validate_client_content(raw.get("content"))
        total_length += content_text_length(content)
        if total_length > MAX_TOTAL_TEXT_LENGTH:
            raise RequestFault("Az üzenetlista teljes tartalma túl nagy.")
        validated.append({"role": role, "content": content})
        last_role = role
    if last_role != "user":
        raise RequestFault("A legutolsó üzenetnek felhasználói feladatnak kell lennie.")
    return validated


def parse_conversation_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value if LOOSE_UUID_PATTERN.fullmatch(value) is not None else None