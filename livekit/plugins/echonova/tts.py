from __future__ import annotations

import asyncio
import json
import time
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from contextlib import aclosing
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectOptions,
    APIError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGiven, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from ._api import (
    SOCKET_ENDED,
    api_errors,
    auth_headers,
    connect_ws,
    event_error,
    read_envelope,
    resolve_api_key,
    resolve_base_url,
    socket_error,
    ws_url,
)
from .log import logger
from .models import TTSNumSteps

SERVICE = "echonova tts"

# https://echonova.ai/docs/reference/tts-stream — the stream sends mono PCM16 at 24 kHz
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
BYTES_PER_SECOND = SAMPLE_RATE * NUM_CHANNELS * 2

# limits on https://echonova.ai/docs/reference/tts-stream
MAX_TEXT_LENGTH = 500
SPEED_RANGE = (0.5, 2.0)
CFG_RANGE = (1.0, 5.0)
NUM_STEPS = (16, 32, 48, 64, 128)

# a socket unused for longer is replaced: below the server's 120 s idle timeout, so none is closed under us
POOL_IDLE_LIMIT = 100.0
# the server ends a session at 30 minutes; a socket is retired this long before
SESSION_SECONDS = 30 * 60
SESSION_MARGIN = 60.0
# a cancel that cannot be written in this long leaves its socket closed instead of reused
CANCEL_SEND_TIMEOUT = 1.0
_CANCEL = json.dumps({"type": "cancel"})
# the SDK's own retries wait this long before the first one
FIRST_RETRY_DELAY = 0.1


def _check_range(name: str, value: float | None, bounds: tuple[float, float]) -> None:
    if value is not None and not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"echonova: {name} must be between {bounds[0]} and {bounds[1]}, got {value}")


def _length(text: str) -> int:
    """Length as the API counts it, in UTF-16 code units."""
    return len(text.encode("utf-16-le")) // 2


def _words(text: str) -> Iterator[str]:
    """The words of `text`, a word over the limit cut into slices that fit."""
    step = MAX_TEXT_LENGTH // 2  # fits even when every code point is a surrogate pair
    for word in text.split():
        if _length(word) <= MAX_TEXT_LENGTH:
            yield word
        else:
            yield from (word[i : i + step] for i in range(0, len(word), step))


def _split(text: str) -> list[str]:
    """Split text at whitespace into pieces the API accepts."""
    text = text.strip()
    if _length(text) <= MAX_TEXT_LENGTH:
        return [text] if text else []

    pieces: list[str] = []
    current = ""
    for word in _words(text):
        candidate = f"{current} {word}" if current else word
        if current and _length(candidate) > MAX_TEXT_LENGTH:
            pieces.append(current)
            current = word
        else:
            current = candidate
    pieces.append(current)
    return pieces


@dataclass(frozen=True)
class VoiceSettings:
    """Per-request voice settings. `None` keeps EchoNova's default."""

    speed: float | None = None  # 0.5-2, default 1
    cfg: float | None = None  # 1-5, how closely the voice follows the text, default 2
    num_steps: TTSNumSteps | int | None = None  # diffusion steps: 16, 32, 48, 64 or 128, default 48
    tashkeel: bool | None = None  # add Arabic vowel marks first, Fusha voices only, default false
    normalize: bool | None = None  # read numbers, dates and money as the dialect says them, default true

    def __post_init__(self) -> None:
        _check_range("speed", self.speed, SPEED_RANGE)
        _check_range("cfg", self.cfg, CFG_RANGE)
        if self.num_steps is not None and self.num_steps not in NUM_STEPS:
            raise ValueError(f"echonova: num_steps must be one of {NUM_STEPS}, got {self.num_steps}")

    def payload(self) -> dict[str, Any]:
        fields = {
            "speed": self.speed,
            "cfg": self.cfg,
            "numSteps": self.num_steps,
            "tashkeel": self.tashkeel,
            "normalize": self.normalize,
        }
        return {k: v for k, v in fields.items() if v is not None}


@dataclass
class _TTSOptions:
    voice_id: str
    voice_settings: VoiceSettings
    api_key: str
    base_url: str
    ssl: bool

    def synthesize_msg(self, text: str, request_id: str) -> str:
        msg = {
            "type": "synthesize",
            "requestId": request_id,
            "text": text,
            "voiceId": self.voice_id,
            **self.voice_settings.payload(),
        }
        return json.dumps(msg, ensure_ascii=False)


def _emit_now(output_emitter: tts.AudioEmitter, data: bytes) -> None:
    """Push PCM and flush it, so the emitter never holds audio back between bursts."""
    output_emitter.push(data)
    output_emitter.flush()


class _SocketPool:
    """The TTS's stream sockets, kept warm across replies: a fresh socket's first synthesis is slow."""

    def __init__(self, open_socket: Callable[[], Awaitable[aiohttp.ClientWebSocketResponse]]) -> None:
        self._open_socket = open_socket
        # reused while used every POOL_IDLE_LIMIT, so a busy socket lasts the call
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect,
            close_cb=self._close,
            max_session_duration=POOL_IDLE_LIMIT,
            mark_refreshed_on_get=True,
        )
        self._opened_at: dict[aiohttp.ClientWebSocketResponse, float] = {}
        self._cancelling: set[asyncio.Task[None]] = set()

    @property
    def last_acquire_time(self) -> float:
        return self._pool.last_acquire_time

    @property
    def last_connection_reused(self) -> bool:
        return self._pool.last_connection_reused

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def acquire(self, conn_options: APIConnectOptions) -> aiohttp.ClientWebSocketResponse:
        """A pooled socket, or a new one; never one the server is about to end at its session cap."""
        if self._cancelling:  # an interrupted socket is warm again as soon as its cancel is written
            await asyncio.wait(self._cancelling)
        while True:
            ws = await self._pool.get(timeout=conn_options.timeout)
            if time.monotonic() - self._opened_at.get(ws, time.monotonic()) < SESSION_SECONDS - SESSION_MARGIN:
                return ws
            self._pool.remove(ws)

    def release(self, ws: aiohttp.ClientWebSocketResponse, *, settled: bool, failed: bool) -> None:
        if failed:
            self._pool.remove(ws)
        elif settled:
            self._pool.put(ws)
        else:  # interrupted mid-synthesis: keep the warm socket
            task = asyncio.create_task(self._cancel(ws))
            self._cancelling.add(task)
            task.add_done_callback(self._cancelling.discard)

    async def aclose(self) -> None:
        await utils.aio.gracefully_cancel(*self._cancelling)
        await self._pool.aclose()

    async def _connect(self, connect_timeout: float) -> aiohttp.ClientWebSocketResponse:
        """The pool's connect callback: the pool hands it the time it may take."""
        with api_errors(f"failed to connect to {SERVICE}"):
            async with asyncio.timeout(connect_timeout):
                ws = await self._open_socket()
        self._opened_at[ws] = time.monotonic()
        return ws

    async def _close(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        self._opened_at.pop(ws, None)
        await ws.close()

    async def _cancel(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Cancel an interrupted synthesis and reuse its socket at once.

        The server finishes the chunk it is generating before it stops, and waiting for that only
        delays the next reply: the server answers that reply after it anyway, and a request skips
        the frames and events of an earlier one.
        """
        reusable = False
        try:
            async with asyncio.timeout(CANCEL_SEND_TIMEOUT):
                await ws.send_str(_CANCEL)
            reusable = True
        except (aiohttp.ClientError, OSError) as e:  # OSError covers the timeout
            logger.debug("echonova tts closed a socket it could not cancel on: %s", e)
        finally:
            if reusable:
                self._pool.put(ws)
            else:
                self._pool.remove(ws)


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str,
        voice_settings: VoiceSettings | None = None,
        api_key: NotGivenOr[str | None] = NOT_GIVEN,
        base_url: NotGivenOr[str | None] = NOT_GIVEN,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer | None] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        ssl: bool = True,
    ) -> None:
        """Create a new instance of EchoNova TTS over the `/v1/text-to-speech/stream` websocket.

        Args:
            voice_id: A voice from `GET /v1/text-to-speech/voices` (see `list_voices`). The
                voice fixes the language and dialect.
            voice_settings: Speed, cfg, diffusion steps, tashkeel and normalization.
            api_key: EchoNova API key, or `ECHONOVA_API_KEY` in the environment.
            base_url: API root; defaults to `ECHONOVA_BASE_URL` or https://api.echonova.ai.
                The websocket url is derived from it.
            tokenizer: Splits streamed text into sentences, one synthesis each.
            http_session: Session to reuse instead of the agent's shared one.
            ssl: Verify TLS certificates.
        """
        super().__init__(
            # sentence-level timed transcripts
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        if not voice_id:
            raise ValueError("echonova: voice_id is required, see GET /v1/text-to-speech/voices")

        self._opts = _TTSOptions(
            voice_id=voice_id,
            voice_settings=voice_settings or VoiceSettings(),
            api_key=resolve_api_key(api_key),
            base_url=resolve_base_url(base_url),
            ssl=ssl,
        )
        self._sentence_tokenizer = (
            tokenizer
            if is_given(tokenizer) and tokenizer is not None
            else tokenize.blingfire.SentenceTokenizer(retain_format=True)
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sockets = _SocketPool(self._open_socket)

    @property
    def model(self) -> str:
        return "echonova"

    @property
    def provider(self) -> str:
        return "echonova"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _open_socket(self) -> aiohttp.ClientWebSocketResponse:
        return await connect_ws(
            self._ensure_session(),
            ws_url(self._opts.base_url, "/v1/text-to-speech/stream"),
            self._opts.api_key,
            service=SERVICE,
            ssl=self._opts.ssl,
        )

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        voice_settings: NotGivenOr[VoiceSettings] = NOT_GIVEN,
    ) -> None:
        """Apply to streams created from here on."""
        if not isinstance(voice_id, NotGiven):
            self._opts.voice_id = voice_id
        if not isinstance(voice_settings, NotGiven):
            self._opts.voice_settings = voice_settings

    async def list_voices(self) -> list[dict[str, Any]]:
        """`GET /v1/text-to-speech/voices`: the voice library and the workspace's cloned voices."""
        with api_errors("echonova voice list failed"):
            async with self._ensure_session().get(
                f"{self._opts.base_url}/v1/text-to-speech/voices",
                headers=auth_headers(self._opts.api_key),
                timeout=aiohttp.ClientTimeout(total=DEFAULT_API_CONNECT_OPTIONS.timeout),
                ssl=self._opts.ssl,
            ) as res:
                data = await read_envelope(res)
        return list(data.get("items") or [])

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> ChunkedStream:
        return ChunkedStream(
            tts=self, input_text=text, conn_options=conn_options, sockets=self._sockets, opts=replace(self._opts)
        )

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> SynthesizeStream:
        stream = SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            sockets=self._sockets,
            opts=replace(self._opts),
            tokenizer=self._sentence_tokenizer,
        )
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        self._sockets.prewarm()

    async def aclose(self) -> None:
        for stream in self._streams.copy():  # a snapshot: aclose() awaits
            await stream.aclose()
        self._streams.clear()
        await self._sockets.aclose()
        await super().aclose()


def _progress(data: str, request_id: str) -> str | None:
    """What a text message reports for `request_id`: "start", "done" or None; raises on a failure."""
    try:
        event = json.loads(data)
    except ValueError:
        raise APIError("echonova tts sent a message that is not JSON", body=data) from None
    if not isinstance(event, dict):
        return None

    kind = event.get("type")
    if event.get("requestId") not in (None, request_id):
        return None  # an interrupted request's late events
    if kind == "error":
        raise event_error(event, SERVICE)
    if kind == "cancelled":
        raise APIError("echonova tts cancelled the synthesis", body=event)
    if kind == "start" and event.get("sampleRate") != SAMPLE_RATE:
        raise APIError(
            f"echonova tts sent {event.get('sampleRate')} Hz, expected {SAMPLE_RATE}", body=event, retryable=False
        )
    return kind if kind in ("start", "done") else None


async def _frames(
    ws: aiohttp.ClientWebSocketResponse, request_id: str, conn_options: APIConnectOptions
) -> AsyncGenerator[bytes, None]:
    """The PCM16 of `request_id` until its done; frames before its start belong to an earlier request."""
    started = False
    while True:
        msg = await ws.receive(timeout=conn_options.timeout)
        if msg.type == aiohttp.WSMsgType.BINARY and started:
            yield msg.data
        elif msg.type == aiohttp.WSMsgType.TEXT:
            progress = _progress(msg.data, request_id)
            if progress == "done":
                return
            started = started or progress == "start"
        elif msg.type in SOCKET_ENDED:
            raise socket_error(ws, msg, SERVICE)


class _PooledSynthesis:
    """One synthesis at a time on the TTS's socket pool, shared by both stream kinds."""

    _sockets: _SocketPool
    _opts: _TTSOptions
    _conn_options: APIConnectOptions
    _acquire_time: float
    _connection_reused: bool

    async def _audio(self, text: str, *, first: bool) -> AsyncGenerator[bytes, None]:
        """The PCM16 of one synthesis, as it arrives.

        The socket goes back to the pool after `done`, and after an interruption once its
        cancel is written, so the next request finds it warm. A socket that failed is closed.
        """
        sockets = self._sockets
        request_id = utils.shortuuid()
        ws = await sockets.acquire(self._conn_options)
        if first:  # the acquisition the stream's first audio waited on
            self._acquire_time = sockets.last_acquire_time
            self._connection_reused = sockets.last_connection_reused
        sent = settled = failed = False
        try:
            with api_errors("echonova tts connection failed"):
                await ws.send_str(self._opts.synthesize_msg(text, request_id))
                sent = True
                async with aclosing(_frames(ws, request_id, self._conn_options)) as frames:
                    async for data in frames:
                        yield data
            settled = True
        except APIError:
            failed = True
            raise
        finally:
            sockets.release(ws, settled=settled or not sent, failed=failed)


class ChunkedStream(_PooledSynthesis, tts.ChunkedStream):
    """The whole text over the stream socket, in pieces of at most 500 characters."""

    def __init__(
        self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions, sockets: _SocketPool, opts: _TTSOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._sockets = sockets
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        for index, piece in enumerate(_split(self._input_text)):
            async with aclosing(self._audio(piece, first=index == 0)) as audio:
                async for data in audio:
                    _emit_now(output_emitter, data)
        output_emitter.flush()


class SynthesizeStream(_PooledSynthesis, tts.SynthesizeStream):
    """One synthesis per sentence, in order: the socket takes one text at a time."""

    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
        sockets: _SocketPool,
        opts: _TTSOptions,
        tokenizer: tokenize.SentenceTokenizer,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._sockets = sockets
        self._opts = opts
        self._tokenizer = tokenizer

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=utils.shortuuid())
        sent_stream = self._tokenizer.stream()

        async def _tokenize_input() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                else:
                    sent_stream.push_text(data)
            sent_stream.end_input()

        async def _synthesize() -> None:
            duration = 0.0
            async for ev in sent_stream:
                pieces = _split(ev.token)
                if not pieces:
                    continue
                self._mark_started()
                output_emitter.push_timed_transcript(TimedString(text=ev.token, start_time=duration))
                for piece in pieces:
                    pushed = await self._speak_piece(piece, output_emitter, retry=duration > 0)
                    duration += pushed / BYTES_PER_SECOND

        tasks = [asyncio.create_task(_tokenize_input()), asyncio.create_task(_synthesize())]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await sent_stream.aclose()

    async def _speak_piece(self, text: str, output_emitter: tts.AudioEmitter, *, retry: bool) -> int:
        """Stream one piece into the emitter; returns its PCM bytes.

        Once audio has played the framework no longer retries, so with `retry` a piece
        that failed before pushing any audio of its own is sent again; without it the
        framework retries the whole stream.
        """
        attempt = 0
        while True:
            pushed = 0
            try:
                async with aclosing(self._audio(text, first=not retry)) as audio:
                    async for data in audio:
                        _emit_now(output_emitter, data)
                        pushed += len(data)
                return pushed
            except APIError as e:
                if not (retry and pushed == 0 and e.retryable and attempt < self._conn_options.max_retry):
                    raise
                interval = FIRST_RETRY_DELAY if attempt == 0 else self._conn_options.retry_interval
                logger.warning("echonova tts piece failed: %s, retrying in %ss", e, interval)
                await asyncio.sleep(interval)
                attempt += 1
