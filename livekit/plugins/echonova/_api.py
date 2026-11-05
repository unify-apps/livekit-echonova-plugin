from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import aiohttp

from livekit.agents import APIConnectionError, APIError, APIStatusError, APITimeoutError
from livekit.agents.types import NotGivenOr
from livekit.agents.utils import is_given

# https://echonova.ai/docs/developers/base-url
BASE_URL = "https://api.echonova.ai"
API_KEY_ENV = "ECHONOVA_API_KEY"
BASE_URL_ENV = "ECHONOVA_BASE_URL"

# a spending-limit 429 holds until the next billing cycle; rate and stream-count 429s clear
_SPENDING_LIMIT = "spending limit"

# https://echonova.ai/docs/developers/errors — the streams' handshake errors
_HANDSHAKE_REASONS = {
    400: "invalid connection parameters",
    401: "invalid or disabled API key",
    402: "workspace out of credits",
    403: "key without access to this service, or subscription inactive",
    503: "streaming not enabled",
}
_NO_RETRY_429 = "spending limit reached, or 10 streams already open in the workspace"
# 1008/1009 bad frame, 4002 no credits, 4029 spending limit
_FATAL_CLOSE_CODES = frozenset({1008, 1009, 4002, 4029})
# the only error a retry can fix
_RETRYABLE_ERRORS = frozenset({"upstream_error"})
SOCKET_ENDED = (
    aiohttp.WSMsgType.CLOSE,
    aiohttp.WSMsgType.CLOSED,
    aiohttp.WSMsgType.CLOSING,
    aiohttp.WSMsgType.ERROR,
)


def resolve_api_key(api_key: NotGivenOr[str | None]) -> str:
    key = (api_key if is_given(api_key) else None) or os.environ.get(API_KEY_ENV)
    if not key:
        raise ValueError(f"echonova: API key required. Set {API_KEY_ENV} or pass api_key.")
    return key


def resolve_base_url(base_url: NotGivenOr[str | None]) -> str:
    url = (base_url if is_given(base_url) else None) or os.environ.get(BASE_URL_ENV) or BASE_URL
    return url.rstrip("/")


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def ws_url(base_url: str, path: str) -> str:
    return base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + path


def status_error(status: int, message: str, *, body: object = None, request_id: str | None = None) -> APIStatusError:
    retryable = False if status == 429 and _SPENDING_LIMIT in message.lower() else None
    return APIStatusError(message, status_code=status, request_id=request_id, body=body, retryable=retryable)


async def read_envelope(res: aiohttp.ClientResponse) -> dict[str, Any]:
    """The `data` of a successful response; any other answer raises its `APIStatusError`."""
    try:
        body: Any = await res.json(content_type=None)
    except ValueError:  # not JSON
        body = None

    if isinstance(body, dict) and 200 <= res.status < 300 and body.get("success", True):
        data = body.get("data")
        return data if isinstance(data, dict) else {}

    envelope = body if isinstance(body, dict) else {}
    status = res.status if res.status >= 400 else int(envelope.get("statusCode") or 502)
    message = envelope.get("errorMessage") or res.reason or "echonova request failed"
    raise status_error(status, str(message), body=body)


def handshake_error(e: aiohttp.WSServerHandshakeError, service: str) -> APIStatusError:
    """A refused upgrade carries no body, so the status and Retry-After are all there is to go on."""
    retry_after = e.headers.get("Retry-After") if e.headers else None
    reason = _HANDSHAKE_REASONS.get(e.status, "unexpected status")
    if e.status == 429:
        reason = f"connection rate limit, retry after {retry_after} s" if retry_after else _NO_RETRY_429
    return APIStatusError(
        f"{service} refused the connection ({e.status}: {reason})",
        status_code=e.status,
        # only the connection-rate 429 clears by waiting; it alone sends Retry-After
        retryable=False if e.status == 503 or (e.status == 429 and not retry_after) else None,
    )


@contextmanager
def api_errors(failure: str) -> Iterator[None]:
    """Raise what fails inside as the APIError the SDK retries on."""
    try:
        yield
    except APIError:
        raise
    except TimeoutError:
        raise APITimeoutError() from None
    except (aiohttp.ClientError, OSError) as e:
        raise APIConnectionError(failure) from e


async def connect_ws(
    session: aiohttp.ClientSession, url: str, api_key: str, *, service: str, ssl: bool, heartbeat: float | None = None
) -> aiohttp.ClientWebSocketResponse:
    """Open a stream socket; the caller bounds how long that may take."""
    with api_errors(f"failed to connect to {service}"):
        try:
            return await session.ws_connect(url, headers=auth_headers(api_key), ssl=ssl, heartbeat=heartbeat)
        except aiohttp.WSServerHandshakeError as e:
            raise handshake_error(e, service) from None


def event_error(event: dict[str, Any], service: str) -> APIError:
    code = event.get("code")
    return APIError(f"{service} {code}: {event.get('message', '')}", body=event, retryable=code in _RETRYABLE_ERRORS)


def socket_error(ws: aiohttp.ClientWebSocketResponse, msg: aiohttp.WSMessage, service: str) -> APIError:
    """The error for a socket that ended while it should still be open."""
    if msg.type == aiohttp.WSMsgType.ERROR:
        error = APIConnectionError(f"{service} connection lost")
        error.__cause__ = ws.exception()
        return error

    code = ws.close_code or -1
    return APIError(
        f"{service} connection closed ({code})",
        body=f"{msg.data=} {msg.extra=}",
        retryable=code not in _FATAL_CLOSE_CODES,
    )
