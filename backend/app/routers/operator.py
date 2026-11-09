from __future__ import annotations

import logging
import math
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect

from app import database as database_module
from app.agent import cognitive as cog
from app.database import DatabaseError
from app.errors import PayloadTooLargeFault, RateLimitFault, RequestFault
from app.utils.access import Access, job_access, load_access
from app.utils.rate_limit import client_key, rate_limiter
from app.utils.security import is_loose_uuid, verify_origin
from app.utils.validation import parse_json_body, read_request_body

router = APIRouter()
logger = logging.getLogger("app.operator")

MAX_OPERATOR_MESSAGE_LENGTH = 32_000
MAX_OPERATOR_MESSAGE_BYTES = 128_000
MAX_BODY_BYTES = 524_288
DEFAULT_RETRY_AFTER = 60
MIN_RETRY_AFTER = 1
MAX_RETRY_AFTER = 86_400
PROTECTED_HEADERS = frozenset({"cache-control", "retry-after"})


def _build_headers(
    access: Access | None, extra_headers: dict[str, str] | None = None
) -> dict[str, str]:
    headers: dict[str, str] = {}
    if access is not None:
        try:
            access_headers = dict(access.headers())
        except Exception:
            logger.exception("operator_access_headers_failed")
            access_headers = {}
        for key, value in access_headers.items():
            name = str(key).strip().lower()
            if not name or name in PROTECTED_HEADERS:
                continue
            headers[name] = str(value)
    headers["cache-control"] = "no-store"
    if extra_headers:
        for key, value in extra_headers.items():
            headers[str(key).strip().lower()] = str(value)
    return headers


def _json(
    payload: dict[str, Any],
    status_code: int = 200,
    access: Access | None = None,
    extra_headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=payload,
        headers=_build_headers(access, extra_headers),
    )


def _error(
    message: str,
    status_code: int,
    access: Access | None = None,
    extra_headers: dict[str, str] | None = None,
) -> JSONResponse:
    return _json({"error": message}, status_code, access, extra_headers)


def _retry_after(exc: RateLimitFault) -> str:
    raw = getattr(exc, "retry_after", None)
    if raw is None or isinstance(raw, bool):
        logger.warning(
            "operator_retry_after_missing",
            extra={"retry_after_type": type(raw).__name__},
        )
        return str(DEFAULT_RETRY_AFTER)
    try:
        seconds = float(raw)
    except (TypeError, ValueError, OverflowError):
        logger.warning(
            "operator_retry_after_invalid",
            extra={"retry_after_type": type(raw).__name__},
        )
        return str(DEFAULT_RETRY_AFTER)
    if math.isnan(seconds):
        logger.warning(
            "operator_retry_after_invalid",
            extra={"retry_after_type": type(raw).__name__},
        )
        return str(DEFAULT_RETRY_AFTER)
    if math.isinf(seconds):
        return str(MAX_RETRY_AFTER if seconds > 0 else MIN_RETRY_AFTER)
    return str(min(MAX_RETRY_AFTER, max(MIN_RETRY_AFTER, math.ceil(seconds))))


def _rate_limited(exc: RateLimitFault, access: Access) -> JSONResponse:
    return _error(exc.message, 429, access, {"Retry-After": _retry_after(exc)})


@router.post("/api/operator")
async def operator_message_endpoint(request: Request) -> JSONResponse:
    try:
        loaded_access = load_access(request)
    except (PayloadTooLargeFault, RequestFault) as exc:
        return _error(exc.message, exc.status_code, None)
    except Exception:
        logger.exception("operator_access_failed")
        return _error("Belső kiszolgálói hiba.", 500, None)

    if loaded_access is None:
        logger.error("operator_access_missing")
        return _error("Belső kiszolgálói hiba.", 500, None)

    access: Access = loaded_access
    client_id = access.client_id

    try:
        rate_limiter.check("operator", client_key(request, client_id))
        origin_error = verify_origin(request)
        if origin_error is not None:
            return _error(origin_error, 403, access)
    except RateLimitFault as exc:
        return _rate_limited(exc, access)
    except (PayloadTooLargeFault, RequestFault) as exc:
        return _error(exc.message, exc.status_code, access)
    except Exception:
        logger.exception("operator_request_failed", extra={"stage": "guard"})
        return _error("Belső kiszolgálói hiba.", 500, access)

    try:
        raw = await read_request_body(request, MAX_BODY_BYTES)
        body = parse_json_body(raw)
    except (PayloadTooLargeFault, RequestFault) as exc:
        return _error(exc.message, exc.status_code, access)
    except ClientDisconnect:
        return _error("A kapcsolat megszakadt.", 400, access)
    except (RecursionError, ValueError):
        return _error("Hibás JSON.", 400, access)
    except Exception:
        logger.exception("operator_request_failed", extra={"stage": "body"})
        return _error("Belső kiszolgálói hiba.", 500, access)

    if not isinstance(body, dict):
        return _error("A kérés törzsének JSON objektumnak kell lennie.", 400, access)

    raw_task_id = body.get("task_id")
    content = body.get("content")

    if not isinstance(raw_task_id, str) or not is_loose_uuid(raw_task_id):
        return _error("Hibás task_id.", 400, access)

    try:
        task_id = str(uuid.UUID(raw_task_id.strip()))
    except (ValueError, AttributeError, TypeError):
        return _error("Hibás task_id.", 400, access)

    if content is None:
        return _error("Hiányzó üzenet.", 400, access)
    if not isinstance(content, str):
        return _error("Az üzenet csak szöveg lehet.", 400, access)

    message = content.strip()
    if not message:
        return _error("Üres üzenet.", 400, access)
    if "\x00" in message:
        return _error("Az üzenet érvénytelen karaktereket tartalmaz.", 400, access)
    if len(message) > MAX_OPERATOR_MESSAGE_LENGTH:
        return _error("Az üzenet túl hosszú.", 413, access)
    try:
        encoded_message = message.encode("utf-8")
    except UnicodeEncodeError:
        return _error("Az üzenet érvénytelen karaktereket tartalmaz.", 400, access)
    if len(encoded_message) > MAX_OPERATOR_MESSAGE_BYTES:
        return _error("Az üzenet túl hosszú.", 413, access)

    try:
        database = database_module.get_admin_database()
        job = await job_access(database, task_id)
    except DatabaseError as exc:
        logger.warning(
            "operator_job_lookup_failed",
            extra={"task_id": task_id, "error_type": type(exc).__name__},
        )
        return _error("Az adatbázis nem érhető el.", 503, access)
    except Exception:
        logger.exception("operator_job_lookup_failed", extra={"task_id": task_id})
        return _error("Belső kiszolgálói hiba.", 500, access)

    if job is None or not job.exists:
        return _error("A feladat nem található.", 404, access)
    if not client_id or not job.owned_by(client_id):
        return _error("A feladat nem található.", 404, access)

    try:
        await cog.post_operator_message(task_id, message)
    except (PayloadTooLargeFault, RequestFault) as exc:
        return _error(exc.message, exc.status_code, access)
    except DatabaseError as exc:
        logger.warning(
            "operator_message_failed",
            extra={"task_id": task_id, "error_type": type(exc).__name__},
        )
        return _error("Mentési hiba.", 503, access)
    except Exception:
        logger.exception("operator_message_failed", extra={"task_id": task_id})
        return _error("Mentési hiba.", 500, access)

    return _json({"accepted": True}, 202, access)