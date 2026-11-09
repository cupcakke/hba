from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from importlib import import_module, metadata
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.routing import Match, Route
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket

from app.config import get_settings
from app.database import close_databases
from app.middleware.error_handler import register_error_handlers
from app.middleware.security import RequestContextMiddleware
from app.routers import browser as browser_router
from app.routers import chat as chat_router
from app.routers import chat_jobs as chat_jobs_router
from app.routers import conversations as conversations_router
from app.routers import images as images_router
from app.routers import operator as operator_router
from app.routers import public_worker as public_worker_router
from app.routers import system as system_router
from app.services.supervisor import start_supervisor, stop_supervisor, supervisor_status

logger = logging.getLogger("app")

_PUBLIC_STATIC_EXTENSIONS = frozenset(
    {
        ".aac",
        ".avif",
        ".bmp",
        ".css",
        ".eot",
        ".flac",
        ".gif",
        ".htm",
        ".html",
        ".ico",
        ".jpeg",
        ".jpg",
        ".js",
        ".m4a",
        ".map",
        ".mjs",
        ".mp3",
        ".mp4",
        ".ogg",
        ".ogv",
        ".opus",
        ".otf",
        ".png",
        ".svg",
        ".ttf",
        ".txt",
        ".wasm",
        ".wav",
        ".webm",
        ".webmanifest",
        ".webp",
        ".woff",
        ".woff2",
        ".xml",
    }
)

_PUBLIC_STATIC_FILENAMES = frozenset(
    {
        "asset-manifest.json",
        "manifest.json",
        "robots.txt",
    }
)

_HASHED_ASSET_PATTERN = re.compile(r"[.\-_][0-9a-f]{8,64}\.[a-z0-9]+$")

_DEFAULT_CORS_METHODS = ("DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT")

_DEFAULT_CORS_HEADERS = (
    "Accept",
    "Accept-Language",
    "Authorization",
    "Cache-Control",
    "Content-Language",
    "Content-Type",
    "Last-Event-ID",
    "X-Request-ID",
    "X-Requested-With",
)

_API_FALLBACK_METHODS = (
    "DELETE",
    "GET",
    "HEAD",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
    "TRACE",
)

_WARNED_KEYS: set[str] = set()
_WARNED_KEYS_LOCK = Lock()


@dataclass(frozen=True)
class _ShutdownPolicy:
    grace_seconds: float
    supervisor_grace_seconds: float
    cancel_grace_seconds: float
    server_timeout_seconds: float | None


@dataclass(frozen=True)
class _TaskOutcome:
    error: BaseException | None = None
    cancellation: asyncio.CancelledError | None = None


@dataclass(frozen=True)
class _HealthPolicy:
    max_consecutive_failures: int | None
    max_sweep_age_seconds: float | None


@dataclass(frozen=True)
class _CorsConfiguration:
    origins: list[str]
    allow_credentials: bool
    allow_methods: list[str]
    allow_headers: list[str]
    max_age_seconds: int


def _warn_once(key: str, message: str, *args: Any) -> None:
    with _WARNED_KEYS_LOCK:
        if key in _WARNED_KEYS:
            return
        _WARNED_KEYS.add(key)

    logger.warning(message, *args)


def configure_logging(settings: Any | None = None) -> None:
    settings = get_settings() if settings is None else settings
    configured_level = getattr(settings, "log_level", None)

    if configured_level is None:
        configured_level = "INFO"

    if isinstance(configured_level, bool):
        raise ValueError("log_level must be a logging level name or a nonnegative integer.")

    if isinstance(configured_level, int):
        if configured_level < 0:
            raise ValueError("log_level must not be negative.")
        level = configured_level
    elif isinstance(configured_level, str):
        level_name = configured_level.strip().upper()

        if level_name.isascii() and level_name.isdecimal():
            level = int(level_name)
        else:
            resolved_level = logging.getLevelName(level_name)
            if not isinstance(resolved_level, int):
                raise ValueError(f"Unsupported log_level: {level_name!r}.")
            level = resolved_level
    else:
        raise ValueError("log_level must be a logging level name or a nonnegative integer.")

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger.setLevel(level)
    logger.disabled = False
    logger.propagate = True
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _string_values(value: Any, setting_name: str) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    elif isinstance(value, (set, frozenset)):
        values = list(value)
    else:
        raise ValueError(
            f"{setting_name} must be a comma-separated string or a collection of strings."
        )

    result: list[str] = []
    seen: set[str] = set()

    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"Every value in {setting_name} must be a string.")
        normalized = item.strip()
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)

    return result


def _boolean_setting(value: Any, setting_name: str) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False

    raise ValueError(f"{setting_name} must be a boolean.")


def _boolean_with_default(settings: Any, setting_name: str, default: bool) -> bool:
    value = getattr(settings, setting_name, None)

    if value is None:
        return default

    return _boolean_setting(value, setting_name)


def _header_setting(settings: Any, setting_name: str, default: str | None) -> str | None:
    value = getattr(settings, setting_name, None)

    if value is None:
        return default

    if not isinstance(value, str):
        raise ValueError(f"{setting_name} must be a string.")

    stripped = value.strip()

    if "\r" in stripped or "\n" in stripped:
        raise ValueError(f"{setting_name} must not contain line breaks.")

    return stripped or None


def _normalize_origin(origin: str) -> str:
    if origin == "*":
        return origin

    if "*" in origin or any(character.isspace() for character in origin):
        raise ValueError("CORS origins must be explicit origins without wildcard patterns.")

    try:
        parsed = urlsplit(origin)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("A configured CORS origin is not a valid URL origin.") from exc

    scheme = parsed.scheme.lower()

    if (
        scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "CORS origins must contain only an HTTP or HTTPS scheme, hostname, and optional port."
        )

    try:
        hostname = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("A configured CORS origin contains an invalid hostname.") from exc

    if ":" in hostname:
        hostname = f"[{hostname}]"

    default_port = 80 if scheme == "http" else 443
    port_suffix = "" if port is None or port == default_port else f":{port}"

    return f"{scheme}://{hostname}{port_suffix}"


def _finite_seconds(
    value: Any,
    setting_name: str,
    *,
    allow_zero: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{setting_name} must be a finite number of seconds.")

    try:
        seconds = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{setting_name} must be a finite number of seconds."
        ) from exc

    if not math.isfinite(seconds):
        raise ValueError(f"{setting_name} must be finite.")

    if seconds < 0 or (seconds == 0 and not allow_zero):
        requirement = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{setting_name} must be {requirement}.")

    return seconds


def _cors_configuration(settings: Any) -> _CorsConfiguration:
    configured = _string_values(
        getattr(settings, "app_origins", None),
        "app_origins",
    )

    if not configured:
        configured = _string_values(
            getattr(settings, "app_origin", None),
            "app_origin",
        )

    origins = list(dict.fromkeys(_normalize_origin(origin) for origin in configured))
    allow_credentials = _boolean_with_default(settings, "cors_allow_credentials", False)

    if "*" in origins and allow_credentials:
        raise ValueError(
            "Wildcard CORS origins cannot be combined with credentialed requests."
        )

    if "*" in origins:
        origins = ["*"]

    configured_methods = _string_values(
        getattr(settings, "cors_allow_methods", None),
        "cors_allow_methods",
    )

    if not configured_methods:
        configured_methods = list(_DEFAULT_CORS_METHODS)

    allow_methods = list(dict.fromkeys(method.upper() for method in configured_methods))

    allow_headers = _string_values(
        getattr(settings, "cors_allow_headers", None),
        "cors_allow_headers",
    )

    if not allow_headers:
        allow_headers = list(_DEFAULT_CORS_HEADERS)

    max_age_value = getattr(settings, "cors_max_age_seconds", None)
    max_age_seconds = (
        600
        if max_age_value is None
        else int(_finite_seconds(max_age_value, "cors_max_age_seconds", allow_zero=True))
    )

    return _CorsConfiguration(
        origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        max_age_seconds=max_age_seconds,
    )


def _project_directories() -> tuple[Path, Path, Path]:
    package_directory = Path(__file__).resolve().parent
    backend_directory = package_directory.parent
    repository_directory = backend_directory.parent
    return package_directory, backend_directory, repository_directory


def _load_toml_module() -> Any | None:
    for module_name in ("tomllib", "tomli"):
        try:
            return import_module(module_name)
        except ModuleNotFoundError:
            continue

    return None


def _source_application_version() -> str | None:
    tomllib = _load_toml_module()

    if tomllib is None:
        return None

    _package_directory, backend_directory, repository_directory = _project_directories()

    for directory in (backend_directory, repository_directory):
        project_file = directory / "pyproject.toml"

        try:
            if not project_file.is_file():
                continue
            with project_file.open("rb") as stream:
                project_data = tomllib.load(stream)
        except (OSError, ValueError) as exc:
            logger.warning(
                "application_version_metadata_unreadable path=%s error=%s",
                project_file,
                exc,
                exc_info=True,
            )
            continue

        project = project_data.get("project")
        tool = project_data.get("tool")
        poetry = tool.get("poetry") if isinstance(tool, Mapping) else None

        for section in (project, poetry):
            if not isinstance(section, Mapping):
                continue

            version = section.get("version")
            if isinstance(version, str) and version.strip():
                return version.strip()

            distribution_name = section.get("name")
            if isinstance(distribution_name, str) and distribution_name.strip():
                try:
                    installed_version = metadata.version(distribution_name.strip())
                except metadata.PackageNotFoundError:
                    installed_version = None

                if installed_version:
                    return installed_version

    return None


def _application_version(settings: Any, override: str | None = None) -> str:
    if override is not None:
        if not isinstance(override, str) or not override.strip():
            raise ValueError("The application version must be a nonempty string.")
        return override.strip()

    for setting_name in ("app_version", "api_version", "version"):
        configured = getattr(settings, setting_name, None)
        if configured is None:
            continue
        if not isinstance(configured, str):
            raise ValueError(f"{setting_name} must be a string.")
        if configured.strip():
            return configured.strip()

    distribution_name = getattr(settings, "distribution_name", None)

    if distribution_name is None:
        distribution_name = "chat-pal-backend"

    if not isinstance(distribution_name, str) or not distribution_name.strip():
        raise ValueError("distribution_name must be a nonempty string.")

    try:
        installed_version = metadata.version(distribution_name.strip())
    except metadata.PackageNotFoundError:
        installed_version = None

    if installed_version:
        return installed_version

    package = sys.modules.get(__package__ or "app")
    package_version = getattr(package, "__version__", None)

    if isinstance(package_version, str) and package_version.strip():
        return package_version.strip()

    source_version = _source_application_version()
    if source_version is not None:
        return source_version

    raise RuntimeError(
        "The application version is unavailable. Configure app_version or api_version, "
        "pass version to create_app(), or install the application's distribution metadata."
    )


def _validated_static_directory(candidate: Path) -> Path:
    resolved = candidate.resolve()

    if not resolved.is_dir():
        raise ValueError("The static path is not a directory.")

    entrypoint = resolved / "chat.html"

    if not entrypoint.is_file():
        raise ValueError("The static directory does not contain chat.html.")

    resolved_entrypoint = entrypoint.resolve(strict=True)
    resolved_entrypoint.relative_to(resolved)

    logger.info("static_directory_selected path=%s", resolved)
    return resolved


def resolve_static_directory(settings: Any | None = None) -> Path | None:
    settings = get_settings() if settings is None else settings
    _package_directory, backend_directory, repository_directory = _project_directories()
    configured = getattr(settings, "static_dir", None)

    if isinstance(configured, str):
        configured = configured.strip()

    if configured is not None and configured != "":
        if not isinstance(configured, (str, os.PathLike)):
            raise ValueError("static_dir must be a string or a filesystem path.")

        configured_path = Path(configured)

        if not configured_path.is_absolute():
            configured_path = backend_directory / configured_path

        try:
            return _validated_static_directory(configured_path)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"The configured static_dir {str(configured_path)!r} is not a usable "
                "frontend directory."
            ) from exc

    candidates = [
        repository_directory / "public",
        backend_directory / "public",
    ]
    seen: set[Path] = set()

    for candidate in candidates:
        try:
            resolved = candidate.resolve()

            if resolved in seen:
                continue
            seen.add(resolved)

            return _validated_static_directory(candidate)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning(
                "static_directory_candidate_rejected path=%s error=%s",
                candidate,
                exc,
            )

    logger.warning(
        "static_directory_not_found frontend_enabled=false candidates=%s",
        [str(candidate) for candidate in candidates],
    )
    return None


def _public_static_path(value: str) -> str:
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or ":" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("Static filenames must be safe relative URL paths.")

    parts = value.split("/")

    if any(
        not part
        or part in {".", ".."}
        or part.startswith(".")
        or part == "__pycache__"
        for part in parts
    ):
        raise ValueError("Static filenames cannot contain hidden or traversal components.")

    if parts[0].casefold() == "api":
        raise ValueError("The API namespace cannot be used for static files.")

    return "/".join(parts)


def _static_extension_allowed(relative: str, extra_extensions: frozenset[str]) -> bool:
    path = PurePosixPath(relative)
    suffix = path.suffix.lower()

    return (
        suffix in _PUBLIC_STATIC_EXTENSIONS
        or suffix in extra_extensions
        or path.name in _PUBLIC_STATIC_FILENAMES
    )


def _extra_static_extensions(settings: Any) -> frozenset[str]:
    configured = _string_values(
        getattr(settings, "static_extra_extensions", None),
        "static_extra_extensions",
    )
    result: set[str] = set()

    for extension in configured:
        normalized = extension.lower()

        if not normalized.startswith("."):
            normalized = f".{normalized}"

        if (
            len(normalized) < 2
            or normalized.count(".") != 1
            or "/" in normalized
            or "\\" in normalized
            or any(character.isspace() for character in normalized)
        ):
            raise ValueError(f"Invalid static_extra_extensions entry: {extension!r}.")

        result.add(normalized)

    return frozenset(result)


def _is_regular_file_within(root: Path, relative: str) -> bool:
    try:
        current = root

        for part in relative.split("/"):
            current = current / part
            if current.is_symlink():
                return False

        resolved = current.resolve(strict=True)
        resolved.relative_to(root)

        return resolved.is_file()
    except (OSError, ValueError, RuntimeError):
        return False


def _static_allowlist(
    settings: Any,
    directory: Path,
    extra_extensions: frozenset[str],
) -> frozenset[str] | None:
    configured = getattr(settings, "static_allowed_files", None)

    if configured is None:
        if not _is_regular_file_within(directory, "chat.html"):
            raise ValueError("The static directory must contain a regular chat.html file.")

        logger.info("static_allowlist_dynamic directory=%s", directory)
        return None

    allowed: set[str] = set()

    for filename in _string_values(configured, "static_allowed_files"):
        relative = _public_static_path(filename)

        if not _static_extension_allowed(relative, extra_extensions):
            raise ValueError(
                f"Invalid static_allowed_files entry: {relative!r} has a disallowed extension."
            )

        if not _is_regular_file_within(directory, relative):
            raise ValueError(f"Invalid static_allowed_files entry: {relative!r}.")

        allowed.add(relative)

    if "chat.html" not in allowed:
        raise ValueError("The static asset allowlist must include a regular chat.html file.")

    return frozenset(allowed)


def _static_response_headers(settings: Any) -> dict[str, str]:
    headers: dict[str, str] = {"X-Content-Type-Options": "nosniff"}

    for setting_name, header_name, default in (
        ("static_frame_options", "X-Frame-Options", "DENY"),
        (
            "static_referrer_policy",
            "Referrer-Policy",
            "strict-origin-when-cross-origin",
        ),
        ("static_content_security_policy", "Content-Security-Policy", None),
    ):
        value = _header_setting(settings, setting_name, default)

        if value is not None:
            headers[header_name] = value

    return headers


class _PublicStaticFiles(StaticFiles):
    def __init__(
        self,
        directory: Path,
        explicit_files: frozenset[str] | None,
        extra_extensions: frozenset[str],
        spa_fallback: bool,
        response_headers: Mapping[str, str],
    ) -> None:
        super().__init__(
            directory=str(directory),
            html=False,
            check_dir=True,
            follow_symlink=False,
        )
        self.root_directory = directory
        self.explicit_files = explicit_files
        self.extra_extensions = extra_extensions
        self.spa_fallback = spa_fallback
        self.response_headers = dict(response_headers)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return

        await super().__call__(scope, receive, send)

    def _is_public_file(self, relative: str) -> bool:
        if self.explicit_files is not None:
            if relative not in self.explicit_files:
                return False
        elif not _static_extension_allowed(relative, self.extra_extensions):
            return False

        return _is_regular_file_within(self.root_directory, relative)

    async def get_response(self, path: str, scope: Scope) -> Response:
        if scope.get("method") not in {"GET", "HEAD"}:
            raise HTTPException(
                status_code=405,
                detail="Method Not Allowed",
                headers={"Allow": "GET, HEAD"},
            )

        normalized = path.replace(os.sep, "/")

        try:
            normalized = _public_static_path(normalized)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Not Found") from exc

        is_public = await asyncio.to_thread(self._is_public_file, normalized)

        if not is_public:
            if self.spa_fallback and PurePosixPath(normalized).suffix == "":
                normalized = "chat.html"
            else:
                raise HTTPException(status_code=404, detail="Not Found")

        response = await super().get_response(normalized, scope)

        for header_name, header_value in self.response_headers.items():
            response.headers.setdefault(header_name, header_value)

        if "Cache-Control" not in response.headers:
            suffix = PurePosixPath(normalized).suffix.lower()
            file_name = PurePosixPath(normalized).name.lower()

            if suffix in {".html", ".htm"}:
                response.headers["Cache-Control"] = "no-cache"
            elif _HASHED_ASSET_PATTERN.search(file_name):
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            else:
                response.headers["Cache-Control"] = "no-cache"

        return response


def _shutdown_policy(
    settings: Any,
    grace_override: float | None = None,
    server_timeout_override: float | None = None,
) -> _ShutdownPolicy:
    grace_value = (
        grace_override
        if grace_override is not None
        else getattr(settings, "shutdown_grace_seconds", None)
    )
    if grace_value is None:
        grace_value = 15.0

    requested_grace = _finite_seconds(grace_value, "shutdown_grace_seconds")
    grace_seconds = requested_grace

    server_value = server_timeout_override
    if server_value is None:
        server_value = getattr(settings, "server_shutdown_timeout_seconds", None)
    if server_value is None:
        server_value = getattr(settings, "asgi_shutdown_timeout_seconds", None)

    server_timeout: float | None = None

    if server_value is not None:
        server_timeout = _finite_seconds(
            server_value,
            "server_shutdown_timeout_seconds",
        )
        margin_value = getattr(settings, "shutdown_timeout_margin_seconds", None)
        if margin_value is None:
            margin_value = min(1.0, server_timeout / 10.0)

        margin = _finite_seconds(
            margin_value,
            "shutdown_timeout_margin_seconds",
            allow_zero=True,
        )

        if margin >= server_timeout:
            raise ValueError(
                "shutdown_timeout_margin_seconds must be smaller than the server shutdown timeout."
            )

        grace_seconds = min(requested_grace, server_timeout - margin)

        if grace_seconds < requested_grace:
            logger.info(
                "shutdown_grace_capped requested_seconds=%s effective_seconds=%s",
                requested_grace,
                grace_seconds,
            )
    else:
        logger.warning(
            "server_shutdown_timeout_unspecified "
            "configure_server_shutdown_timeout_seconds_to_match_the_ASGI_server=true"
        )

    cancel_value = getattr(settings, "shutdown_cancel_grace_seconds", None)
    if cancel_value is None:
        cancel_value = min(1.0, grace_seconds / 10.0)

    cancel_grace = _finite_seconds(
        cancel_value,
        "shutdown_cancel_grace_seconds",
        allow_zero=True,
    )

    usable_seconds = grace_seconds - 2.0 * cancel_grace

    if usable_seconds <= 0:
        raise ValueError(
            "shutdown_cancel_grace_seconds must leave time for cleanup within the "
            "effective shutdown grace."
        )

    database_value = getattr(settings, "database_shutdown_grace_seconds", None)
    if database_value is None:
        database_value = min(5.0, usable_seconds / 3.0)

    database_grace = _finite_seconds(
        database_value,
        "database_shutdown_grace_seconds",
        allow_zero=True,
    )

    if database_grace >= usable_seconds:
        raise ValueError(
            "database_shutdown_grace_seconds must be smaller than the effective shutdown grace "
            "after reserving time for cancellation."
        )

    return _ShutdownPolicy(
        grace_seconds=grace_seconds,
        supervisor_grace_seconds=usable_seconds - database_grace,
        cancel_grace_seconds=cancel_grace,
        server_timeout_seconds=server_timeout,
    )


def _health_policy(settings: Any) -> _HealthPolicy:
    failures_value = getattr(settings, "health_max_consecutive_failures", None)
    max_failures: int | None

    if failures_value is None:
        max_failures = 10
    elif (
        isinstance(failures_value, bool)
        or not isinstance(failures_value, int)
        or failures_value < 0
    ):
        raise ValueError("health_max_consecutive_failures must be a nonnegative integer.")
    else:
        max_failures = failures_value or None

    age_value = getattr(settings, "health_max_sweep_age_seconds", None)
    max_age = (
        None
        if age_value is None
        else _finite_seconds(age_value, "health_max_sweep_age_seconds")
    )

    return _HealthPolicy(
        max_consecutive_failures=max_failures,
        max_sweep_age_seconds=max_age,
    )


def _release_abandoned_task(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return

    error = task.exception()

    if error is not None:
        logger.error(
            "abandoned_task_failed task=%s",
            task.get_name(),
            exc_info=(type(error), error, error.__traceback__),
        )


async def _drain_task(
    task: asyncio.Task[Any],
    timeout: float | None = None,
) -> _TaskOutcome:
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + max(0.0, timeout)
    cancellation: asyncio.CancelledError | None = None

    while not task.done():
        remaining = None if deadline is None else deadline - loop.time()

        if remaining is not None and remaining <= 0:
            break

        try:
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
                logger.warning(
                    "shutdown_cancellation_deferred task=%s",
                    task.get_name(),
                )

    if not task.done():
        task.add_done_callback(_release_abandoned_task)
        logger.error(
            "task_abandoned task=%s timeout_seconds=%s",
            task.get_name(),
            timeout,
        )
        return _TaskOutcome(
            error=TimeoutError(
                f"Task {task.get_name()!r} did not finish within {timeout} seconds."
            ),
            cancellation=cancellation,
        )

    try:
        task.result()
    except BaseException as exc:
        return _TaskOutcome(error=exc, cancellation=cancellation)

    return _TaskOutcome(cancellation=cancellation)


async def _run_cleanup_phase(
    operation: Callable[[], Awaitable[Any]],
    operation_name: str,
    grace_seconds: float,
    cancel_grace_seconds: float,
) -> _TaskOutcome:
    async def invoke() -> None:
        await operation()

    loop = asyncio.get_running_loop()
    task = asyncio.create_task(invoke(), name=operation_name)
    deadline = loop.time() + max(0.0, grace_seconds)
    cancellation: asyncio.CancelledError | None = None

    while not task.done():
        remaining = deadline - loop.time()

        if remaining <= 0:
            break

        try:
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc

    timed_out = not task.done()

    if timed_out:
        logger.warning(
            "%s_grace_exceeded grace_seconds=%s cancelling_cleanup=true",
            operation_name,
            grace_seconds,
        )
        task.cancel()

    outcome = await _drain_task(task, cancel_grace_seconds if timed_out else None)
    error = outcome.error

    if timed_out and (error is None or isinstance(error, asyncio.CancelledError)):
        error = TimeoutError(
            f"{operation_name} exceeded its {grace_seconds} second grace period "
            "and was cancelled."
        )

    return _TaskOutcome(
        error=error,
        cancellation=(
            cancellation
            if cancellation is not None
            else outcome.cancellation
        ),
    )


async def _shutdown_resources(
    application: FastAPI,
    policy: _ShutdownPolicy,
) -> None:
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    deadline = started_at + policy.grace_seconds
    outcomes: list[tuple[str, _TaskOutcome]] = []

    operations: tuple[
        tuple[str, Callable[[], Awaitable[Any]]],
        tuple[str, Callable[[], Awaitable[Any]]],
    ] = (
        ("supervisor_stop", stop_supervisor),
        ("database_close", close_databases),
    )

    try:
        for operation_name, operation in operations:
            grace_seconds = (
                policy.supervisor_grace_seconds
                if operation_name == "supervisor_stop"
                else max(0.0, deadline - loop.time() - policy.cancel_grace_seconds)
            )

            try:
                outcome = await _run_cleanup_phase(
                    operation,
                    operation_name,
                    grace_seconds,
                    policy.cancel_grace_seconds,
                )
            except BaseException as exc:
                outcome = _TaskOutcome(error=exc)

            outcomes.append((operation_name, outcome))

            if outcome.error is not None:
                error = outcome.error
                logger.error(
                    "%s_failed",
                    operation_name,
                    exc_info=(type(error), error, error.__traceback__),
                )
    finally:
        elapsed = loop.time() - started_at
        complete = len(outcomes) == len(operations) and all(
            outcome.error is None for _operation_name, outcome in outcomes
        )
        application.state.shutdown_complete = complete

        if (
            policy.server_timeout_seconds is not None
            and elapsed >= policy.server_timeout_seconds
        ):
            logger.error(
                "application_shutdown_exceeded_server_budget "
                "elapsed_seconds=%s server_timeout_seconds=%s",
                elapsed,
                policy.server_timeout_seconds,
            )

        logger.info(
            "application_stopped cleanup_complete=%s elapsed_seconds=%s",
            complete,
            elapsed,
        )

    errors = [
        outcome.error
        for _operation_name, outcome in outcomes
        if outcome.error is not None
    ]

    for error in errors:
        if not isinstance(error, (Exception, asyncio.CancelledError)):
            raise error

    for _operation_name, outcome in outcomes:
        if outcome.cancellation is not None:
            raise outcome.cancellation

    for error in errors:
        if isinstance(error, asyncio.CancelledError):
            raise error

    if errors:
        raise RuntimeError(
            f"Application shutdown failed in {len(errors)} cleanup operation(s)."
        ) from errors[0]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    policy: _ShutdownPolicy = application.state.shutdown_policy
    primary_error: BaseException | None = None

    application.state.shutting_down = False
    application.state.startup_complete = False
    application.state.shutdown_complete = False

    logger.info("application_starting")

    try:
        try:
            await start_supervisor()
        except asyncio.CancelledError:
            logger.warning("supervisor_start_cancelled", exc_info=True)
            raise
        except Exception:
            logger.exception("supervisor_start_failed")
            raise

        application.state.startup_complete = True
        logger.info("application_started")
        yield
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        application.state.shutting_down = True
        application.state.startup_complete = False

        shutdown_task = asyncio.create_task(
            _shutdown_resources(application, policy),
            name="application_shutdown",
        )
        outcome = await _drain_task(
            shutdown_task,
            policy.grace_seconds + policy.cancel_grace_seconds,
        )

        if outcome.error is not None:
            error = outcome.error

            if primary_error is not None:
                logger.error(
                    "application_shutdown_failed_during_existing_error",
                    exc_info=(type(error), error, error.__traceback__),
                )
            elif outcome.cancellation is None:
                raise error

        if outcome.cancellation is not None:
            raise outcome.cancellation


def _status_flag(status: Mapping[str, Any], name: str) -> bool | None:
    value = status.get(name)

    if value is None or isinstance(value, bool):
        return value

    _warn_once(
        f"status_flag_{name}",
        "supervisor_status_field_ignored field=%s reason=not_boolean",
        name,
    )
    return None


def _status_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}

    if not isinstance(value, Mapping):
        _warn_once(
            f"status_mapping_{name}",
            "supervisor_status_field_ignored field=%s reason=not_mapping",
            name,
        )
        return {}

    return value


def _status_failures(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _warn_once(
            "status_consecutive_failures",
            "supervisor_status_field_ignored field=consecutive_failures "
            "reason=not_nonnegative_integer",
        )
        return None

    return value


def _public_timestamp(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, (str, int, float, date, datetime)):
        _warn_once(
            "status_last_sweep_at_type",
            "supervisor_status_field_ignored field=last_sweep_at reason=unsupported_type",
        )
        return None

    if isinstance(value, float) and not math.isfinite(value):
        _warn_once(
            "status_last_sweep_at_finite",
            "supervisor_status_field_ignored field=last_sweep_at reason=not_finite",
        )
        return None

    return jsonable_encoder(value)


def _parse_sweep_time(value: Any) -> datetime | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None

        try:
            parsed = datetime.fromtimestamp(value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str):
        text = value.strip()

        if not text:
            return None

        if text.endswith(("Z", "z")):
            text = f"{text[:-1]}+00:00"

        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def _supervisor_health(
    policy: _HealthPolicy,
) -> tuple[bool, dict[str, Any], list[str]]:
    status = supervisor_status()

    if not isinstance(status, Mapping):
        raise TypeError("supervisor_status() must return a mapping.")

    enabled = _status_flag(status, "enabled")
    running = _status_flag(status, "running")
    leader = _status_flag(status, "leader")
    breakers = _status_mapping(status.get("breakers"), "breakers")
    metrics = _status_mapping(status.get("metrics"), "metrics")
    open_breakers: list[str] = []

    for name, breaker in breakers.items():
        if not isinstance(name, str) or not name:
            _warn_once(
                "status_breaker_name",
                "supervisor_breaker_ignored reason=invalid_name",
            )
            continue

        if not isinstance(breaker, Mapping):
            _warn_once(
                f"status_breaker_value_{name}",
                "supervisor_breaker_ignored name=%s reason=not_mapping",
                name,
            )
            continue

        state = breaker.get("state")

        if not isinstance(state, str) or not state.strip():
            _warn_once(
                f"status_breaker_state_{name}",
                "supervisor_breaker_ignored name=%s reason=invalid_state",
                name,
            )
            continue

        if state.strip().lower() == "open":
            open_breakers.append(name)

    consecutive_failures = _status_failures(status.get("consecutive_failures"))
    raw_last_sweep_at = metrics.get("last_sweep_at")
    last_sweep_at = _public_timestamp(raw_last_sweep_at)
    parsed_sweep_time = _parse_sweep_time(raw_last_sweep_at)
    sweep_age = (
        None
        if parsed_sweep_time is None
        else max(0.0, (datetime.now(timezone.utc) - parsed_sweep_time).total_seconds())
    )

    stalled = enabled is True and running is not True and leader is not False

    failures_exceeded = (
        policy.max_consecutive_failures is not None
        and consecutive_failures is not None
        and consecutive_failures >= policy.max_consecutive_failures
    )

    sweep_stale = (
        policy.max_sweep_age_seconds is not None
        and enabled is True
        and running is True
        and leader is not False
        and sweep_age is not None
        and sweep_age > policy.max_sweep_age_seconds
    )

    reasons: list[str] = []

    if open_breakers:
        reasons.append("open_breakers")
    if stalled:
        reasons.append("supervisor_stalled")
    if failures_exceeded:
        reasons.append("consecutive_failures_exceeded")
    if sweep_stale:
        reasons.append("sweep_stale")

    public_status = {
        "available": True,
        "enabled": enabled,
        "running": running,
        "leader": leader,
        "consecutive_failures": consecutive_failures,
        "open_breakers": sorted(open_breakers),
        "last_sweep_at": last_sweep_at,
        "last_sweep_age_seconds": None if sweep_age is None else round(sweep_age, 3),
        "stale": sweep_stale,
    }

    return not reasons, public_status, reasons


def _validate_reserved_routes(application: FastAPI) -> None:
    for reserved_path in ("/", "/api/health"):
        matching_routes = [
            route
            for route in application.router.routes
            if isinstance(route, Route)
            and route.path == reserved_path
            and (route.methods is None or "GET" in route.methods)
        ]

        if len(matching_routes) != 1:
            raise RuntimeError(
                f"The reserved GET route {reserved_path!r} must be defined exactly once. "
                "Remove the conflicting route from the included router."
            )


def create_app(
    *,
    settings: Any | None = None,
    version: str | None = None,
    shutdown_grace_seconds: float | None = None,
    server_shutdown_timeout_seconds: float | None = None,
) -> FastAPI:
    settings = get_settings() if settings is None else settings
    configure_logging(settings)

    cors = _cors_configuration(settings)
    application_version = _application_version(settings, version)
    policy = _shutdown_policy(
        settings,
        grace_override=shutdown_grace_seconds,
        server_timeout_override=server_shutdown_timeout_seconds,
    )
    health_policy = _health_policy(settings)
    docs_enabled = _boolean_with_default(settings, "api_docs_enabled", True)

    static_directory = resolve_static_directory(settings)
    static_files: _PublicStaticFiles | None = None

    if static_directory is not None:
        extra_extensions = _extra_static_extensions(settings)
        static_files = _PublicStaticFiles(
            directory=static_directory,
            explicit_files=_static_allowlist(settings, static_directory, extra_extensions),
            extra_extensions=extra_extensions,
            spa_fallback=_boolean_with_default(settings, "static_spa_fallback", False),
            response_headers=_static_response_headers(settings),
        )

    app = FastAPI(
        title="Chat Pal Backend",
        version=application_version,
        docs_url="/api/docs" if docs_enabled else None,
        openapi_url="/api/openapi.json" if docs_enabled else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    app.state.shutdown_policy = policy
    app.state.shutting_down = False
    app.state.startup_complete = False
    app.state.shutdown_complete = False
    app.state.frontend_available = static_files is not None

    register_error_handlers(app)

    @app.get("/api/health", name="health")
    async def health() -> JSONResponse:
        shutting_down = bool(app.state.shutting_down)

        try:
            healthy, public_status, reasons = _supervisor_health(health_policy)
        except Exception:
            logger.exception("supervisor_status_failed")

            unavailable_reasons = ["supervisor_status_unavailable"]

            if shutting_down:
                unavailable_reasons.append("shutting_down")

            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "service": "chat-pal-backend",
                    "reasons": unavailable_reasons,
                    "supervisor": {
                        "available": False,
                        "enabled": None,
                        "running": None,
                        "leader": None,
                        "consecutive_failures": None,
                        "open_breakers": [],
                        "last_sweep_at": None,
                        "last_sweep_age_seconds": None,
                        "stale": None,
                    },
                },
                headers={"Cache-Control": "no-store"},
            )

        if shutting_down:
            healthy = False
            reasons = [*reasons, "shutting_down"]

        return JSONResponse(
            status_code=200 if healthy else 503,
            content={
                "status": "ok" if healthy else "degraded",
                "service": "chat-pal-backend",
                "reasons": reasons,
                "supervisor": public_status,
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/", include_in_schema=False, name="frontend_index")
    async def index(request: Request) -> Response:
        if static_files is None:
            raise HTTPException(
                status_code=404,
                detail="The frontend is not configured.",
            )

        target = request.url_for("static", path="chat.html")

        return RedirectResponse(
            url=target.path,
            status_code=307,
            headers={"Cache-Control": "no-store"},
        )

    app.include_router(chat_router.router)
    app.include_router(chat_jobs_router.router)
    app.include_router(conversations_router.router)
    app.include_router(operator_router.router)
    app.include_router(browser_router.router)
    app.include_router(images_router.router)
    app.include_router(public_worker_router.router)
    app.include_router(system_router.router)

    _validate_reserved_routes(app)

    known_routes = tuple(app.router.routes)

    async def api_not_found(request: Request) -> Response:
        allowed_methods: set[str] = set()

        for route in known_routes:
            match, _child_scope = route.matches(request.scope)

            if match is Match.PARTIAL:
                methods = getattr(route, "methods", None)
                if methods:
                    allowed_methods.update(methods)

        if allowed_methods:
            raise HTTPException(
                status_code=405,
                detail="Method Not Allowed",
                headers={"Allow": ", ".join(sorted(allowed_methods))},
            )

        raise HTTPException(status_code=404, detail="Not Found")

    async def api_websocket_not_found(websocket: WebSocket) -> None:
        await websocket.close(code=1008)

    app.router.add_route(
        "/api",
        api_not_found,
        methods=_API_FALLBACK_METHODS,
        name="api_not_found_root",
        include_in_schema=False,
    )
    app.router.add_route(
        "/api/{path:path}",
        api_not_found,
        methods=_API_FALLBACK_METHODS,
        name="api_not_found",
        include_in_schema=False,
    )
    app.router.add_websocket_route(
        "/api",
        api_websocket_not_found,
        name="api_websocket_not_found_root",
    )
    app.router.add_websocket_route(
        "/api/{path:path}",
        api_websocket_not_found,
        name="api_websocket_not_found",
    )

    if static_files is not None:
        app.mount("/", static_files, name="static")

    if cors.origins:

        def cors_factory(application: ASGIApp) -> ASGIApp:
            return CORSMiddleware(
                application,
                allow_origins=cors.origins,
                allow_credentials=cors.allow_credentials,
                allow_methods=cors.allow_methods,
                allow_headers=cors.allow_headers,
                expose_headers=["X-Request-ID"],
                max_age=cors.max_age_seconds,
            )

        app.add_middleware(cors_factory)

    def request_context_factory(application: ASGIApp) -> ASGIApp:
        return RequestContextMiddleware(application)

    app.add_middleware(request_context_factory)

    return app


class _LazyApplication:
    def __init__(self, factory: Callable[[], FastAPI]) -> None:
        self._factory = factory
        self._application: FastAPI | None = None
        self._failure: Exception | None = None
        self._lock = Lock()

    def _get_application(self) -> FastAPI:
        application = self._application

        if application is not None:
            return application

        with self._lock:
            application = self._application

            if application is not None:
                return application

            if self._failure is not None:
                raise RuntimeError(
                    "Application initialization failed earlier. "
                    "Fix the configuration and restart the process."
                ) from self._failure

            try:
                application = self._factory()
            except Exception as exc:
                self._failure = exc
                logger.exception("application_initialization_failed")
                raise RuntimeError("Application initialization failed.") from exc

            self._application = application
            return application

    @property
    def application(self) -> FastAPI:
        return self._get_application()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        return getattr(self._get_application(), name)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            application = self._get_application()
        except Exception:
            if scope["type"] == "lifespan":
                message = await receive()

                if message["type"] == "lifespan.startup":
                    await send(
                        {
                            "type": "lifespan.startup.failed",
                            "message": (
                                "Application initialization failed. "
                                "Consult the server logs for details."
                            ),
                        }
                    )
                    return

            raise

        await application(scope, receive, send)


app = _LazyApplication(create_app)