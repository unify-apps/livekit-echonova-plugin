from __future__ import annotations

import asyncio
import json
import time
import weakref
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast
from urllib.parse import urlencode

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIError,
    LanguageCode,
    inference,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGiven, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.vad import VAD, VADEventType, VADStream

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
from .models import STTLanguages

SERVICE = "echonova stt"
DEFAULT_LANGUAGE = "ar"
LANGUAGES = ("ar", "en")

# https://echonova.ai/docs/reference/stt-transcribe-sync — up to 10 minutes of audio per call
TRANSCRIBE_TIMEOUT = 60.0

# https://echonova.ai/docs/reference/stt-stream
DEFAULT_SAMPLE_RATE = 16000
SAMPLE_RATE_RANGE = (8000, 48000)
FRAME_SECONDS = 0.1  # audio per binary frame, as the docs recommend
# the docs recommend 300-700 ms of silence to end a segment: the low end, as it adds to every reply
DEFAULT_MIN_SILENCE = 0.3
# audio kept while nobody speaks and sent when speech starts: covers the VAD's prefix padding
PRE_ROLL_SECONDS = 1.0
# the most unjudged audio kept, should a VAD stop reporting its progress
MAX_BACKLOG_SECONDS = 10.0
# https://echonova.ai/docs/reference/stt-stream — every text frame, the terminology one included
MAX_FRAME_BYTES = 64 * 1024
# the server closes a socket that gets no frame for 120 s
KEEPALIVE_INTERVAL = 30.0
# a socket is replaced during silence this long before the server's session cap; above the keepalive,
# since a silent socket is only checked when it pings
SESSION_MARGIN = 60.0
DEFAULT_MAX_SESSION_SECONDS = 1500

# session.ended reasons a new socket clears
_RETRYABLE_ENDINGS = frozenset({"max_duration", "idle_timeout", "upstream_error"})

Terminology = Sequence[str | Mapping[str, Any]]

_END_SEGMENT = json.dumps({"type": "end_segment"})
_STOP = json.dumps({"type": "stop"})
_PING = json.dumps({"type": "ping"})
_INPUT_ENDED = object()  # queued after the last audio: the socket is told to stop


def _language(value: str | None) -> str:
    """The EchoNova language for a language tag: "ar-AE" is "ar"; anything but ar or en raises."""
    code = LanguageCode(value).language if value else DEFAULT_LANGUAGE
    if code not in LANGUAGES:
        raise ValueError(f"echonova: language must be one of {LANGUAGES}, got {value!r}")
    return code


@dataclass(frozen=True)
class TranscriptionOptions:
    """What EchoNova is asked for. `None` keeps its default."""

    # names, products and acronyms to spell your way: each a term, or {"term": ..., "variants": [...]}
    terminology: Terminology | None = None
    numbers_as_digits: bool | None = None  # write spoken Arabic numbers as digits; recognize() only
    sample_rate: int = DEFAULT_SAMPLE_RATE  # 8000-48000 Hz the stream is sent at
    # ends each streamed segment at a pause; None leaves that to flush()
    vad: NotGivenOr[VAD | None] = NOT_GIVEN
    # seconds of silence that end a segment, for the default VAD: in STT turn detection, the turn's end
    min_silence: float = DEFAULT_MIN_SILENCE

    def __post_init__(self) -> None:
        if not SAMPLE_RATE_RANGE[0] <= self.sample_rate <= SAMPLE_RATE_RANGE[1]:
            raise ValueError(f"echonova: sample_rate must be between 8000 and 48000, got {self.sample_rate}")
        if self.min_silence <= 0:
            raise ValueError(f"echonova: min_silence must be above 0, got {self.min_silence}")


@dataclass
class _STTOptions:
    language: str
    terminology: Terminology | None
    numbers_as_digits: bool | None
    sample_rate: int
    vad: VAD | None
    api_key: str
    base_url: str
    ssl: bool

    @property
    def transcribe_url(self) -> str:
        return f"{self.base_url}/v1/speech-to-text/sync"

    @property
    def stream_url(self) -> str:
        query = urlencode({"sample_rate": self.sample_rate, "language": self.language})
        return ws_url(self.base_url, f"/v1/speech-to-text/stream?{query}")

    def terminology_msg(self) -> str:
        return _terminology_msg(self.terminology)

    def form(self, wav: bytes, language: str) -> aiohttp.FormData:
        form = aiohttp.FormData()
        form.add_field("file", wav, filename="audio.wav", content_type="audio/wav")
        form.add_field("language", language)
        if self.numbers_as_digits is not None:
            form.add_field("numbers_as_digits", str(self.numbers_as_digits).lower())
        if self.terminology:
            form.add_field("terminology", json.dumps({"keywords": list(self.terminology)}, ensure_ascii=False))
        return form


def _terminology_msg(terminology: Terminology | None) -> str:
    return json.dumps({"type": "terminology", "terminology": {"keywords": list(terminology or ())}}, ensure_ascii=False)


def _check_terminology(terminology: Terminology | None) -> None:
    """The stream takes terminology in one frame, so a list that can't fit fails here, not mid-call."""
    size = len(_terminology_msg(terminology).encode())
    if terminology and size > MAX_FRAME_BYTES:
        raise ValueError(f"echonova: terminology is {size} bytes, over the stream's {MAX_FRAME_BYTES}-byte frame")


def _event(data: str) -> dict[str, Any]:
    try:
        event = json.loads(data)
    except ValueError:
        logger.warning("echonova stt sent a message that is not JSON")
        return {}
    return event if isinstance(event, dict) else {}


def _transcript(event: Mapping[str, Any]) -> str:
    """The segment's text, corrected to the terminology when terms are set."""
    corrected = event.get("terminology")
    if isinstance(corrected, Mapping) and corrected.get("corrected"):
        return str(corrected["corrected"])
    return str(event.get("text") or "")


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: STTLanguages | str | None = DEFAULT_LANGUAGE,
        streaming: bool = True,
        options: TranscriptionOptions | None = None,
        api_key: NotGivenOr[str | None] = NOT_GIVEN,
        base_url: NotGivenOr[str | None] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        ssl: bool = True,
    ) -> None:
        """Create a new instance of EchoNova STT.

        Args:
            language: "ar" (Arabic and mixed Arabic/English) or "en" (English only). A
                regional tag such as "ar-AE" uses its language.
            streaming: True streams the call over `/v1/speech-to-text/stream`: audio goes out
                while the caller speaks and each segment's transcript comes back when a pause
                ends it. False sends each utterance to `POST /v1/speech-to-text/sync`, at half
                the price, after the agent's VAD has cut it.
            options: Terminology, numbers as digits, the stream's sample rate and the VAD
                that ends its segments (by default LiveKit's bundled Silero).
            api_key: EchoNova API key, or `ECHONOVA_API_KEY` in the environment.
            base_url: API root; defaults to `ECHONOVA_BASE_URL` or https://api.echonova.ai.
            http_session: Session to reuse instead of the agent's shared one.
            ssl: Verify TLS certificates.

        EchoNova sends no interim results on either path.
        """
        options = options or TranscriptionOptions()
        segmenter = options.vad if is_given(options.vad) else None
        if streaming and not is_given(options.vad):
            segmenter = inference.VAD(model="silero", min_silence_duration=options.min_silence)

        if streaming:
            _check_terminology(options.terminology)
        super().__init__(capabilities=stt.STTCapabilities(streaming=streaming, interim_results=False))
        self._opts = _STTOptions(
            language=_language(language),
            terminology=options.terminology,
            numbers_as_digits=options.numbers_as_digits,
            sample_rate=options.sample_rate,
            vad=segmenter,
            api_key=resolve_api_key(api_key),
            base_url=resolve_base_url(base_url),
            ssl=ssl,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

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

    def update_options(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        numbers_as_digits: NotGivenOr[bool | None] = NOT_GIVEN,
        terminology: NotGivenOr[Terminology | None] = NOT_GIVEN,
    ) -> None:
        """Apply to the next utterance and to streams opened from here on."""
        if not isinstance(language, NotGiven):
            self._opts.language = _language(language)
        if not isinstance(numbers_as_digits, NotGiven):
            self._opts.numbers_as_digits = numbers_as_digits
        if not isinstance(terminology, NotGiven):
            if self.capabilities.streaming:
                _check_terminology(terminology)
            self._opts.terminology = terminology

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        opts = self._opts
        # a per-call language EchoNova can't take falls back to the configured one
        lang = LanguageCode(language).language if is_given(language) and language else opts.language
        if lang not in LANGUAGES:
            lang = opts.language
        wav = rtc.combine_audio_frames(buffer).to_wav_bytes()

        with api_errors("echonova transcription failed"):
            async with self._ensure_session().post(
                opts.transcribe_url,
                data=opts.form(wav, lang),
                headers=auth_headers(opts.api_key),
                timeout=aiohttp.ClientTimeout(total=TRANSCRIBE_TIMEOUT, sock_connect=conn_options.timeout),
                ssl=opts.ssl,
            ) as res:
                data = await read_envelope(res)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=data.get("jobId") or "",
            alternatives=[
                stt.SpeechData(
                    language=LanguageCode(data.get("language") or lang),
                    text=_transcript({"text": data.get("transcription"), "terminology": data.get("terminology")}),
                    start_time=0.0,
                    end_time=float(data.get("durationSeconds") or 0.0),
                )
            ],
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = replace(self._opts)
        if is_given(language) and language:
            opts.language = _language(language)
        stream = SpeechStream(stt=self, opts=opts, conn_options=conn_options, session=self._ensure_session())
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in self._streams.copy():  # a snapshot: aclose() awaits
            await stream.aclose()
        self._streams.clear()
        await super().aclose()


class _Outbox:
    """What the stream's sockets are sent, in order, across every socket of the call.

    While nobody speaks the audio is only kept, as pre-roll. Sent audio is kept until a
    transcript covers it, so a socket that drops loses no speech: the next one is sent it again.
    """

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._chunker = utils.audio.AudioByteStream(
            sample_rate=sample_rate, num_channels=1, samples_per_channel=int(sample_rate * FRAME_SECONDS)
        )
        # (where it ends, in seconds of the call's audio, pcm): trimmed behind what the VAD has judged
        self._pre_roll: deque[tuple[float, bytes]] = deque()
        self._fed = 0.0
        self._judged = 0.0
        self._pending: deque[bytes | str | object] = deque()
        self._unconfirmed: deque[bytes | str] = deque()
        self._ready = asyncio.Event()
        self.forwarding = False
        self.segment_open = False

    def audio(self, pcm: bytes) -> None:
        self._fed += len(pcm) / 2 / self._sample_rate
        if self.forwarding:
            self._put_audio(self._chunker.write(pcm))
            return
        self._pre_roll.append((self._fed, pcm))
        self._trim(self._fed - MAX_BACKLOG_SECONDS)

    @property
    def unjudged(self) -> float:
        """Seconds of fed audio the VAD has not reached yet."""
        return self._fed - self._judged

    def judged(self, seconds: float) -> None:
        """The VAD has judged `seconds` more audio; audio it has not reached yet is never dropped."""
        self._judged += seconds
        self._trim(self._judged - PRE_ROLL_SECONDS)

    def speech_started(self) -> None:
        self.forwarding = True
        while self._pre_roll:
            self._put_audio(self._chunker.write(self._pre_roll.popleft()[1]))

    def _trim(self, before: float) -> None:
        while self._pre_roll and self._pre_roll[0][0] <= before:
            self._pre_roll.popleft()

    def end_segment(self, *, keep_forwarding: bool) -> None:
        self._put_audio(self._chunker.flush())
        if self.segment_open:
            self._put(_END_SEGMENT)
            self.segment_open = False
        self.forwarding = keep_forwarding

    def end_input(self) -> None:
        self.end_segment(keep_forwarding=False)
        self._put(_INPUT_ENDED)

    def resend(self, *, forwarding: bool) -> None:
        """Queue what a lost socket was sent but never confirmed ahead of everything else.

        A new VAD stream follows, and it judges the audio from where the input is now.
        """
        self._judged = self._fed
        self._pending.extendleft(reversed(self._unconfirmed))
        self._unconfirmed.clear()
        if self._pending:
            self._ready.set()
        if not forwarding:  # the new VAD stream starts in silence: close what the old one left open
            self.end_segment(keep_forwarding=False)
        self.forwarding = forwarding

    async def next(self) -> bytes | str | object:
        """The next message to send, once there is one."""
        while not self._pending:
            self._ready.clear()
            await self._ready.wait()
        return self._pending.popleft()

    def sent(self, msg: bytes | str) -> None:
        if isinstance(msg, bytes) or msg == _END_SEGMENT:
            self._unconfirmed.append(msg)

    def confirm(self, seconds: float) -> None:
        """A transcript covered the oldest `seconds` of sent audio."""
        remaining = round(seconds * self._sample_rate) * 2
        while self._unconfirmed and remaining > 0:
            msg = self._unconfirmed.popleft()
            if isinstance(msg, bytes):
                remaining -= len(msg)
        if self._unconfirmed and self._unconfirmed[0] == _END_SEGMENT:
            self._unconfirmed.popleft()

    def _put(self, msg: bytes | str | object) -> None:
        self._pending.append(msg)
        self._ready.set()

    def _put_audio(self, frames: list[rtc.AudioFrame]) -> None:
        for frame in frames:
            self._put(frame.data.tobytes())
            self.segment_open = True


class SpeechStream(stt.SpeechStream):
    """One call over `/v1/speech-to-text/stream`, on as many sockets as its length needs.

    Only speech is sent, with a short pre-roll, since EchoNova charges every second it
    receives; the VAD's end of speech ends the segment. Without a VAD everything is sent
    and `flush()` ends a segment.
    """

    def __init__(
        self, *, stt: STT, opts: _STTOptions, conn_options: APIConnectOptions, session: aiohttp.ClientSession
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = session
        self._speaking = False
        self._speech_started_at = 0.0
        # (start, end) of each segment the VAD ended, in this stream's time, for its transcript
        self._segment_times: deque[tuple[float, float]] = deque()
        # outlives a retry, so the audio a failed socket held reaches the next one
        self._outbox = _Outbox(opts.sample_rate)

    async def _run(self) -> None:
        self._outbox.resend(forwarding=self._opts.vad is None)
        self._segment_times.clear()  # resent audio is transcribed without them
        vad_stream = self._opts.vad.stream() if self._opts.vad else None
        feeders = [asyncio.create_task(self._feed(vad_stream))]
        if vad_stream is not None:
            feeders.append(asyncio.create_task(self._segment(vad_stream)))
        try:
            while not await self._serve_socket(feeders):
                logger.debug("echonova stt replaced its socket before the session cap")
        finally:
            await utils.aio.gracefully_cancel(*feeders)
            if vad_stream is not None:
                await vad_stream.aclose()
            self._end_speaking()

    async def _feed(self, vad_stream: VADStream | None) -> None:
        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                if vad_stream is not None:
                    vad_stream.push_frame(data)
                self._outbox.audio(data.data.tobytes())
            else:  # flush: the caller ends the segment
                self._outbox.end_segment(keep_forwarding=vad_stream is None or self._speaking)
        if vad_stream is None:
            self._outbox.end_input()
        else:
            vad_stream.end_input()  # _segment ends the input once the VAD has judged the last frame

    async def _segment(self, vad_stream: VADStream) -> None:
        async for ev in vad_stream:
            if ev.type == VADEventType.START_OF_SPEECH:
                self._outbox.speech_started()
                self._start_speaking(self._when(ev.speech_duration))
            elif ev.type == VADEventType.INFERENCE_DONE:
                self._outbox.judged(sum(frame.duration for frame in ev.frames))
            elif ev.type == VADEventType.END_OF_SPEECH:
                self._outbox.end_segment(keep_forwarding=False)
                ended = self._when(ev.silence_duration)
                self._segment_times.append((self._stream_time(self._speech_started_at), self._stream_time(ended)))
                self._end_speaking(ended)
        self._outbox.end_input()

    async def _serve_socket(self, feeders: list[asyncio.Task[None]]) -> bool:
        """Serve one socket; True once the input is done, False when it was replaced at the session cap."""
        with api_errors("echonova stt connection failed"):
            async with asyncio.timeout(self._conn_options.timeout):
                ws = await connect_ws(
                    self._session,
                    self._opts.stream_url,
                    self._opts.api_key,
                    service=SERVICE,
                    ssl=self._opts.ssl,
                    heartbeat=30.0,
                )
            try:
                return await self._converse(ws, feeders)
            finally:
                await ws.close()

    async def _converse(self, ws: aiohttp.ClientWebSocketResponse, feeders: list[asyncio.Task[None]]) -> bool:
        max_seconds = await self._session_started(ws)
        if self._opts.terminology:
            await ws.send_str(self._opts.terminology_msg())
        writer = asyncio.create_task(self._write(ws, time.monotonic() + max_seconds - SESSION_MARGIN))
        reader = asyncio.create_task(self._read(ws))
        try:
            # a finished feeder is checked once per socket, so its failure is never missed
            pending: set[asyncio.Task[Any]] = {writer, reader, *feeders}
            while not reader.done():
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()  # a failure anywhere ends the socket
            return writer.result() if writer.done() else False
        finally:
            await utils.aio.gracefully_cancel(writer, reader)

    async def _session_started(self, ws: aiohttp.ClientWebSocketResponse) -> int:
        """Wait for session.started, the go-ahead for audio; returns the session's length in seconds."""
        msg = await ws.receive(timeout=self._conn_options.timeout)
        if msg.type in SOCKET_ENDED:
            raise socket_error(ws, msg, SERVICE)
        try:
            event = json.loads(msg.data) if msg.type == aiohttp.WSMsgType.TEXT else {}
        except ValueError:
            event = {}
        if event.get("type") == "error":
            raise event_error(event, SERVICE)
        if event.get("type") != "session.started":
            raise APIError(f"{SERVICE} expected session.started, got {event.get('type')!r}", body=event)
        return int(event.get("maxSessionSeconds") or DEFAULT_MAX_SESSION_SECONDS)

    async def _write(self, ws: aiohttp.ClientWebSocketResponse, rotate_at: float) -> bool:
        """Send the outbox; True once the input's stop went out, False after stopping at the session cap."""
        while (msg := await self._next_message()) is not _INPUT_ENDED:
            # unconfirmed before the send, so a message a failing send loses is sent again
            self._outbox.sent(cast(bytes | str, msg))
            if isinstance(msg, bytes):
                await ws.send_bytes(msg)
            else:
                await ws.send_str(cast(str, msg))
            if not self._outbox.segment_open and time.monotonic() >= rotate_at:
                await ws.send_str(_STOP)
                return False
        await ws.send_str(_STOP)
        return True

    async def _next_message(self) -> bytes | str | object:
        """The outbox's next message, or a ping when there is none for a while, to keep the socket open."""
        try:
            async with asyncio.timeout(KEEPALIVE_INTERVAL):
                return await self._outbox.next()
        except TimeoutError:
            return _PING

    async def _read(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Emit each segment's transcript until session.ended answers our stop."""
        while True:
            msg = await ws.receive()
            if msg.type in SOCKET_ENDED:
                raise socket_error(ws, msg, SERVICE)
            if msg.type == aiohttp.WSMsgType.TEXT and self._on_event(_event(msg.data)):
                return

    def _on_event(self, event: dict[str, Any]) -> bool:
        """Handle one server event; True once session.ended answers our stop."""
        kind = event.get("type")
        if kind == "transcript":
            self._on_transcript(event)
        elif kind == "session.ended":
            if event.get("reason") == "stopped":
                return True
            raise APIError(
                f"{SERVICE} session ended: {event.get('reason')}",
                body=event,
                retryable=event.get("reason") in _RETRYABLE_ENDINGS,
            )
        elif kind == "error" and event.get("code") != "validation_failed":
            raise event_error(event, SERVICE)
        elif kind == "error":  # an invalid terminology list is ignored and the session goes on
            logger.warning("echonova stt rejected a message: %s", event.get("message"))
        return False

    def _when(self, seconds_before_judged: float) -> float:
        """Wall-clock time of the audio that far behind the VAD's position: the input is live."""
        return time.time() - self._outbox.unjudged - seconds_before_judged

    def _stream_time(self, wall: float) -> float:
        """The SDK's transcript clock, which it maps back to wall-clock time."""
        return self.start_time_offset + wall - self.start_time

    def _on_transcript(self, event: dict[str, Any]) -> None:
        duration = float(event.get("durationSeconds") or 0.0)
        self._outbox.confirm(duration)
        start, end = self._segment_times.popleft() if self._segment_times else (0.0, 0.0)
        if duration > 0:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    recognition_usage=stt.RecognitionUsage(audio_duration=duration),
                )
            )
        if text := _transcript(event):
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=str(event.get("segmentId") or ""),
                    alternatives=[
                        stt.SpeechData(
                            language=LanguageCode(event.get("language") or self._opts.language),
                            text=text,
                            start_time=start,
                            end_time=end,
                        )
                    ],
                )
            )

    def _start_speaking(self, at: float) -> None:
        if not self._speaking:
            self._speaking = True
            self._speech_started_at = at
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH, speech_start_time=at))

    def _end_speaking(self, at: float | None = None) -> None:
        """`at` is when speech stopped, so the turn's endpointing delay runs from there, not from now."""
        if self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, speech_end_time=at))
