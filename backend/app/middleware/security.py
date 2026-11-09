from __future__ import annotations

import logging
import time
import uuid

logger = logging.getLogger("app.request")

REQUEST_ID_HEADER = "X-Request-Id"


class RequestContextMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = scope.get("headers", [])
        # Extract request_id from headers if present
        headers_dict = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        request_id_val = headers_dict.get(REQUEST_ID_HEADER.lower()) or uuid.uuid4().hex
        scope["state"] = scope.get("state", {})
        # Store request_id in scope state for later use if needed
        started = time.monotonic()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.lower().encode(), request_id_val.encode()))
                duration_ms = int((time.monotonic() - started) * 1000)
                headers.append((b"x-response-time-ms", str(duration_ms).encode()))
                message["headers"] = headers
                # Log after response start? We log based on path
                path = scope.get("path", "")
                method = scope.get("method", "")
                status = message.get("status", 200)
                if path.startswith("/api/"):
                    logger.info(
                        "request",
                        extra={
                            "request_id": request_id_val,
                            "method": method,
                            "path": path,
                            "status": status,
                            "duration_ms": duration_ms,
                        },
                    )
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id_val,
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                },
            )
            raise
