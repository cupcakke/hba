from __future__ import annotations

import asyncio
import json
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from app.agent import cognitive, provider
from app.agent.provider import ModelStream, call_model, receive_model
from app.agent.runtime import AgentRuntime, FATAL_CODES
from app.errors import RequestFault
from app.models import RuntimeConfig
from app.routers.chat import _merge_history
from app.agent import subagents
from app.agent.tools import MODEL_SUBAGENT_TOOLS, enabled_tool_names
from app.errors import ProviderFault
from app.main import create_app
from app.models import TerminalState
from app.services.chat_jobs import DurableEventSink, _retryable
from app.services.supervisor import Supervisor
from app.utils.validation import validate_client_history


class StaticAssetRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_installed_app_assets_are_present_and_valid(self) -> None:
        app = create_app(version="test")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            manifest_response = await client.get("/manifest.webmanifest")
            self.assertEqual(manifest_response.status_code, 200)
            manifest = manifest_response.json()
            self.assertEqual(manifest["display"], "standalone")

            icon_paths = [entry["src"] for entry in manifest["icons"]]
            icon_paths.extend(["/favicon.ico", "/icons/apple-touch-icon.png"])
            for icon_path in icon_paths:
                with self.subTest(icon=icon_path):
                    response = await client.get(icon_path)
                    self.assertEqual(response.status_code, 200)
                    if icon_path.endswith(".png"):
                        self.assertTrue(response.content.startswith(b"\x89PNG\r\n\x1a\n"))

            favicon = await client.get("/favicon.ico")
            self.assertEqual(favicon.content[:4], struct.pack("<HH", 0, 1))
            robots = await client.get("/robots.txt")
            self.assertEqual(robots.status_code, 200)
            self.assertIn("Disallow: /api/", robots.text)


class AuxiliaryModelPolicyTests(unittest.TestCase):
    def test_main_model_cannot_select_subagent_tools(self) -> None:
        available = set(enabled_tool_names("configured", "configured"))
        self.assertTrue(MODEL_SUBAGENT_TOOLS.isdisjoint(available))


class ReportedChatRegressionTests(unittest.TestCase):
    def test_learned_skill_dataclass_is_read_without_mapping_access(self) -> None:
        runtime = AgentRuntime.__new__(AgentRuntime)
        skill = cognitive.Skill(
            id="skill-id",
            name="learned",
            dsl="SKILL learned\nWHEN useful\nDO act",
            terms="learned useful",
            vector=[],
            successes=1,
            failures=0,
        )

        result = runtime.knowledge_text({"skills": [skill]}, "")

        self.assertIn(skill.dsl, result)

    def test_client_history_accepts_image_and_text_file_content(self) -> None:
        image_url = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        history = validate_client_history(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Nézd meg ezt a képet."},
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {
                                "type": "text_file",
                                "name": "jegyzet.txt",
                                "text": "Teljes csatolt szöveg",
                            },
                        ],
                    }
                ]
            }
        )

        self.assertEqual(history[0]["content"][1]["image_url"]["url"], image_url)
        self.assertIn("Teljes csatolt szöveg", history[0]["content"][2]["text"])

    def test_client_history_rejects_invalid_image_data(self) -> None:
        with self.assertRaises(RequestFault):
            validate_client_history(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,%%%"},
                                }
                            ],
                        }
                    ]
                }
            )

    def test_responses_input_translates_user_images_and_keeps_generated_history_valid(self) -> None:
        image_url = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        converted = provider._responses_input(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Írd le."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Elkészült."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ]
        )

        self.assertEqual(converted[0]["content"][0], {"type": "input_text", "text": "Írd le."})
        self.assertEqual(converted[0]["content"][1]["type"], "input_image")
        self.assertEqual(converted[1]["content"], "Elkészült.\n[A korábbi asszisztensüzenet képet tartalmazott.]")

    def test_history_merge_uses_edited_and_shorter_client_history(self) -> None:
        stored = [
            {"role": "user", "content": "Régi kérés"},
            {"role": "assistant", "content": "Régi válasz"},
            {"role": "user", "content": "Következő kérés"},
        ]
        edited = [{"role": "user", "content": "Szerkesztett kérés"}]
        shortened = stored[:1]

        self.assertEqual(_merge_history(stored, edited), edited)
        self.assertEqual(_merge_history(stored, shortened), shortened)

    def test_browser_hosts_can_coexist_with_sandbox_domains(self) -> None:
        config = RuntimeConfig(
            model_key="configured-model-key",
            browser_hosts=frozenset({"browser.example.com"}),
            allowed_domains=frozenset({"cache.nixos.org"}),
            nixpkgs_rev="a" * 40,
            nix_version="2.24",
            nix_installer_sha256="b" * 64,
        )

        self.assertEqual(config.browser_hosts, frozenset({"browser.example.com"}))
        self.assertEqual(config.allowed_domains, frozenset({"cache.nixos.org"}))


class AdvisorRateLimitRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_advisor_429_suppresses_repeated_requests_during_cooldown(self) -> None:
        attempts = 0

        async def rate_limited_response(*_args, **_kwargs):
            nonlocal attempts
            attempts += 1
            return subagents._HttpOutcome(429, '{"error":"rate limited"}', None)

        with (
            patch.object(subagents, "_api_key", return_value="test-key"),
            patch.object(subagents, "_post_once", new=rate_limited_response),
        ):
            with self.assertRaises(subagents.AdvisorRateLimited):
                await subagents.interact(
                    "test-advisor", "test question", None, fail_fast_on_rate_limit=True
                )
            with self.assertRaises(subagents.AdvisorRateLimited):
                await subagents.interact(
                    "test-advisor", "test question", None, fail_fast_on_rate_limit=True
                )

        self.assertEqual(attempts, 1)


class _FakeSSEResponse:
    headers: dict[str, str] = {}

    def __init__(self, events: list[dict[str, object]]):
        self._body = b"".join(
            f"data: {json.dumps(event)}\n\n".encode("utf-8") for event in events
        )

    async def aiter_bytes(self):
        yield self._body

    async def aclose(self) -> None:
        return None


class _FakeClient:
    def __init__(self):
        self.closed = False
        self.send_count = 0
        self.request_args = ()
        self.request_kwargs = {}

    def build_request(self, *args, **kwargs):
        self.request_args = args
        self.request_kwargs = kwargs
        return object()

    async def send(self, *_args, **_kwargs):
        self.send_count += 1
        return _FakePaymentRequiredResponse()

    async def aclose(self) -> None:
        self.closed = True


class _FakePaymentRequiredResponse:
    status_code = 402
    headers: dict[str, str] = {}

    async def aread(self) -> bytes:
        return b'{"error":"payment required"}'

    async def aclose(self) -> None:
        return None


class ModelReasoningVisibilityTests(unittest.IsolatedAsyncioTestCase):
    async def _receive(self, events: list[dict[str, object]]):
        response = _FakeSSEResponse(events)
        client = _FakeClient()
        stream = ModelStream(response=response, client=client, touch=lambda: None)
        received: list[dict[str, object]] = []

        async def send(event: dict[str, object]) -> None:
            received.append(event)

        result = await receive_model(
            stream, "test-job", 1, send, asyncio.Event()
        )
        return result, received

    async def test_raw_reasoning_deltas_reach_the_visible_event_stream(self) -> None:
        _, events = await self._receive(
            [
                {
                    "type": "response.reasoning_text.delta",
                    "delta": "A provider által küldött gondolatmenet.",
                },
                {"type": "response.output_text.delta", "delta": "Válasz."},
                {
                    "type": "response.completed",
                    "response": {"status": "completed", "output": []},
                },
            ]
        )

        reasoning_events = [
            event for event in events if event.get("reasoning_content")
        ]
        self.assertEqual(len(reasoning_events), 1)
        self.assertEqual(
            reasoning_events[0]["reasoning_content"],
            "A provider által küldött gondolatmenet.",
        )
        self.assertEqual(reasoning_events[0]["reasoning_type"], "raw")

    async def test_buffered_summary_is_shown_when_raw_reasoning_is_unavailable(self) -> None:
        _, events = await self._receive(
            [
                {
                    "type": "response.reasoning_summary_text.delta",
                    "delta": "A provider részletes összefoglalója.",
                },
                {
                    "type": "response.completed",
                    "response": {"status": "completed", "output": []},
                },
            ]
        )

        reasoning_events = [
            event for event in events if event.get("reasoning_content")
        ]
        self.assertEqual(len(reasoning_events), 1)
        self.assertEqual(
            reasoning_events[0]["reasoning_content"],
            "A provider részletes összefoglalója.",
        )
        self.assertEqual(reasoning_events[0]["reasoning_type"], "summary")

    async def test_raw_reasoning_takes_precedence_over_summary(self) -> None:
        _, events = await self._receive(
            [
                {
                    "type": "response.reasoning_summary_text.delta",
                    "delta": "Összefoglaló.",
                },
                {
                    "type": "response.reasoning.delta",
                    "delta": "Nyers gondolatmenet.",
                },
                {
                    "type": "response.completed",
                    "response": {"status": "completed", "output": []},
                },
            ]
        )

        reasoning_events = [
            event for event in events if event.get("reasoning_content")
        ]
        self.assertEqual(len(reasoning_events), 1)
        self.assertEqual(
            reasoning_events[0]["reasoning_content"], "Nyers gondolatmenet."
        )
        self.assertEqual(reasoning_events[0]["reasoning_type"], "raw")

    async def test_raw_reasoning_in_completed_response_is_visible(self) -> None:
        _, events = await self._receive(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output": [
                            {
                                "type": "reasoning",
                                "content": [
                                    {
                                        "type": "reasoning_text",
                                        "text": "A lezáró válaszban küldött reasoning.",
                                    }
                                ],
                            }
                        ],
                    },
                }
            ]
        )

        reasoning_events = [
            event for event in events if event.get("reasoning_content")
        ]
        self.assertEqual(len(reasoning_events), 1)
        self.assertEqual(reasoning_events[0]["reasoning_type"], "raw")
        self.assertEqual(
            reasoning_events[0]["reasoning_content"],
            "A lezáró válaszban küldött reasoning.",
        )


class ModelPaymentRequiredTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_402_is_clear_and_does_not_retry_the_upstream_call(self) -> None:
        fake_client = _FakeClient()
        settings = SimpleNamespace(
            model_name="muse-spark-1.3-contributor",
            model_api_url="https://model.example/v1/responses",
        )
        config = SimpleNamespace(model_key=SimpleNamespace(get_secret_value=lambda: "test"))
        with (
            patch("app.config.get_settings", return_value=settings),
            patch.object(provider.httpx, "AsyncClient", return_value=fake_client),
        ):
            with self.assertRaises(ProviderFault) as caught:
                await call_model(
                    config,
                    [{"role": "user", "content": "Szia"}],
                    [],
                    asyncio.Event(),
                    asyncio.get_running_loop().time() + 30,
                )

        self.assertEqual(caught.exception.code, "MODEL_PAYMENT_REQUIRED")
        self.assertIn("HTTP 402", caught.exception.message)
        self.assertIn("számlázását", caught.exception.message)
        self.assertEqual(fake_client.send_count, 1)
        self.assertTrue(fake_client.closed)
        request_body = json.loads(fake_client.request_kwargs["content"])
        self.assertEqual(request_body["model"], "muse-spark-1.3-contributor")
        self.assertEqual(request_body["max_output_tokens"], 27_000)
        self.assertEqual(request_body["reasoning"]["effort"], "xhigh")

    async def test_payment_required_is_not_retried_by_the_chat_worker(self) -> None:
        self.assertIn("MODEL_PAYMENT_REQUIRED", FATAL_CODES)
        self.assertFalse(
            _retryable(
                TerminalState.FAILED, "failed", "MODEL_PAYMENT_REQUIRED"
            )
        )

    async def test_model_failures_stop_without_runtime_or_job_retries(self) -> None:
        runtime = AgentRuntime.__new__(AgentRuntime)
        runtime.model_failures = 0
        action = runtime.classify_fault(ProviderFault("MODEL_HTTP", "HTTP 503"))

        self.assertEqual(action, "fatal")
        self.assertEqual(runtime.model_failures, 1)
        self.assertFalse(_retryable(TerminalState.FAILED, "failed", "MODEL_HTTP"))
        self.assertFalse(_retryable(TerminalState.FAILED, "failed", "MODEL_RATE_LIMIT"))
        self.assertFalse(_retryable(TerminalState.FAILED, "failed", "PROTOCOL_ERROR"))
        self.assertIn("MODEL_REQUEST_LIMIT", FATAL_CODES)

    async def test_supervisor_does_not_poll_model_api_on_each_health_sweep(self) -> None:
        settings = SimpleNamespace(
            model_api_key="configured",
            instavm_api_key="",
            browser_use_api_key="",
            supervisor_probe_timeout_ms=8_000,
            supervisor_breaker_threshold=4,
            supervisor_breaker_cooldown_ms=120_000,
        )

        async def healthy_database(_supervisor, _client):
            return True, "", 1

        supervisor = Supervisor()
        with (
            patch("app.services.supervisor.get_settings", return_value=settings),
            patch.object(Supervisor, "_probe_database", healthy_database),
        ):
            probes = await supervisor._probe_dependencies()

        self.assertEqual(set(probes), {"database"})

    async def test_reasoning_type_survives_job_progress_storage(self) -> None:
        sink = DurableEventSink("test-job", database=SimpleNamespace())
        sink.close()
        await sink.send(
            {
                "type": "delta",
                "reasoning_content": "raw thought",
                "reasoning_type": "raw",
            }
        )
        await sink.send(
            {
                "type": "error",
                "code": "MODEL_PAYMENT_REQUIRED",
                "message": "HTTP 402",
            }
        )

        self.assertEqual(
            sink.stored_reasoning(),
            "[[chat-reasoning-type:raw]]\nraw thought",
        )
        self.assertEqual(sink.error_code, "MODEL_PAYMENT_REQUIRED")
        self.assertEqual(sink.error_message, "HTTP 402")


if __name__ == "__main__":
    unittest.main()
