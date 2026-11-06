"""Plugin checks against a local fake of the EchoNova API. No key, no network.

pip install . && python tests/test_echonova.py
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from array import array
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Self, cast
from unittest import mock

import aiohttp
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIError, APIStatusError, APITimeoutError
from livekit.agents import stt as lkstt
from livekit.agents.metrics import TTSMetrics
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT
from livekit.agents.vad import VAD, VADEvent, VADEventType
from livekit.plugins import echonova

FAST = APIConnectOptions(max_retry=2, retry_interval=0.05, timeout=2.0)
NO_RETRY = APIConnectOptions(max_retry=0, retry_interval=0.05, timeout=2.0)
FRAME = b"\x01\x00" * 2400  # 100 ms of PCM16 at 24 kHz, every sample 1
FRAMES_PER_SYNTHESIS = 3
SAMPLES = len(FRAME) // 2 * FRAMES_PER_SYNTHESIS  # per synthesis
LATE_FRAME = b"\x02\x00" * 2400  # what a cancelled synthesis still sends: its samples must never be played
SENTENCE_A = "The weather is lovely this morning in the city."
SENTENCE_B = "We will meet at the park by the lake after lunch."
# the fields https://echonova.ai/docs/reference/tts-stream lists for synthesize; any other is refused
SYNTHESIZE_FIELDS = {"type", "requestId", "text", "voiceId", "speed", "cfg", "numSteps", "tashkeel", "normalize"}

STT_RATE = 16000
SPEECH, SILENCE = 7, 0  # the sample value of a speech or a silent 10 ms frame, as the scripted VAD reads it

Behaviour = Callable[[web.WebSocketResponse, dict], Awaitable[bool]]
SpeechEventType = lkstt.SpeechEventType


async def send_start(ws: web.WebSocketResponse, msg: dict, *, rate: int = 24000) -> None:
    start = {"type": "start", "requestId": msg["requestId"], "sampleRate": rate, "encoding": "pcm_s16le", "channels": 1}
    await ws.send_str(json.dumps(start))


async def send_done(ws: web.WebSocketResponse, msg: dict) -> None:
    chars = len(msg["text"])
    done = {"type": "done", "requestId": msg["requestId"], "characters": chars, "creditsConsumed": chars}
    await ws.send_str(json.dumps({**done, "durationSeconds": 0.3}))


async def speak(ws: web.WebSocketResponse, msg: dict, *, rate: int = 24000) -> bool:
    """A normal synthesis: start, frames, done. Returns False to end the connection."""
    await send_start(ws, msg, rate=rate)
    for _ in range(FRAMES_PER_SYNTHESIS):
        await ws.send_bytes(FRAME)
    await send_done(ws, msg)
    return True


async def hang_after_audio(ws: web.WebSocketResponse, msg: dict) -> bool:
    await send_start(ws, msg)
    await ws.send_bytes(FRAME)
    return True  # no done: the client must give up on this socket


async def partial_then_drop(ws: web.WebSocketResponse, msg: dict) -> bool:
    await send_start(ws, msg)
    await ws.send_bytes(FRAME)
    await ws.close(code=1011)
    return False


async def drop(ws: web.WebSocketResponse, _msg: dict) -> bool:
    await ws.close(code=1011)
    return False


def error(code: str) -> Behaviour:
    async def behaviour(ws: web.WebSocketResponse, msg: dict) -> bool:
        await ws.send_str(
            json.dumps({"type": "error", "requestId": msg.get("requestId"), "code": code, "message": code})
        )
        return True

    return behaviour


def close_with(code: int) -> Behaviour:
    async def behaviour(ws: web.WebSocketResponse, _msg: dict) -> bool:
        await ws.close(code=code)
        return False

    return behaviour


async def wrong_rate(ws: web.WebSocketResponse, msg: dict) -> bool:
    return await speak(ws, msg, rate=16000)


@dataclass
class Probe:
    """Set `heard` once the client has every sample of the first frame; `held` says it never did."""

    heard: asyncio.Event = field(default_factory=asyncio.Event)
    held: bool = False

    async def burst_until_heard(self, ws: web.WebSocketResponse, msg: dict) -> bool:
        await send_start(ws, msg)
        await ws.send_bytes(FRAME)
        try:
            await asyncio.wait_for(self.heard.wait(), 1)
        except TimeoutError:
            self.held = True
        await send_done(ws, msg)
        return True


def stt_ended(reason: str) -> Behaviour:
    async def behaviour(ws: web.WebSocketResponse, _msg: dict) -> bool:
        await ws.send_str(json.dumps({"type": "session.ended", "reason": reason, "totalSeconds": 0}))
        await ws.close(code=4002 if reason == "insufficient_balance" else 1011)
        return False

    return behaviour


def vad_event(kind: VADEventType, frame: rtc.AudioFrame | None = None) -> VADEvent:
    frames = [frame] if frame else []
    return VADEvent(type=kind, samples_index=0, timestamp=0.0, speech_duration=0.0, silence_duration=0.0, frames=frames)


class ScriptedVADStream:
    """Judges every frame, as LiveKit's VADs report it; speech starts on a SPEECH frame and ends on SILENCE.

    A lagging stream reports nothing until its input ends, like a VAD running far behind the call.
    """

    def __init__(self, *, lagging: bool) -> None:
        self._events: asyncio.Queue[VADEvent | None] = asyncio.Queue()
        self._held: list[VADEvent] = []
        self._lagging = lagging
        self._speaking = False

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._emit(vad_event(VADEventType.INFERENCE_DONE, frame))
        speaking = frame.data[0] == SPEECH
        if speaking != self._speaking:
            self._speaking = speaking
            self._emit(vad_event(VADEventType.START_OF_SPEECH if speaking else VADEventType.END_OF_SPEECH))

    def _emit(self, event: VADEvent) -> None:
        if self._lagging:
            self._held.append(event)
        else:
            self._events.put_nowait(event)

    def end_input(self) -> None:
        for event in self._held:
            self._events.put_nowait(event)
        self._events.put_nowait(None)

    async def aclose(self) -> None:
        self._events.put_nowait(None)

    def __aiter__(self) -> ScriptedVADStream:
        return self

    async def __anext__(self) -> VADEvent:
        event = await self._events.get()
        if event is None:
            raise StopAsyncIteration
        return event


@dataclass
class ScriptedVAD:
    lagging: bool = False

    def stream(self) -> ScriptedVADStream:
        return ScriptedVADStream(lagging=self.lagging)


@dataclass
class SttSocket:
    """What one /v1/speech-to-text/stream socket received."""

    query: dict[str, str]
    segments: list[bytes] = field(default_factory=lambda: [b""])
    texts: list[dict] = field(default_factory=list)
    order: list[str] = field(default_factory=list)  # message kinds in arrival order


@dataclass
class FakeEchoNova:
    """`plan[i]` answers the i-th synthesize across all sockets; later ones speak normally."""

    plan: list[Behaviour] = field(default_factory=list)
    refuse: tuple[int, dict[str, str]] | None = None  # status and headers for every stream upgrade
    answer_cancel: bool = True  # a cancel gets late frames, then cancelled
    # how an unanswered, interrupted synthesis ends once the next one arrives, as the real stream does
    late_event: str = "done"
    # pings every this many seconds and drops a socket that missed the last one, as the real stream does every 25 s
    ping_interval: float | None = None
    running: str = ""  # the requestId of the last synthesize
    unfinished: str = ""  # a synthesis hang_after_audio left running
    connections: int = 0
    messages: list[dict] = field(default_factory=list)
    stt_forms: list[dict] = field(default_factory=list)
    stt_response: tuple[int, dict] = (200, {})
    stt_delay: float = 0.0
    stt_plan: list[Behaviour] = field(default_factory=list)  # `stt_plan[i]` runs on the i-th stream socket
    stt_max_session: int = 1500
    stt_drop_after_speech: int | None = None  # the first stream socket closes 1011 past this many speech samples
    stt_sockets: list[SttSocket] = field(default_factory=list)
    url: str = ""
    _runner: web.AppRunner | None = None

    @property
    def synthesized(self) -> list[str]:
        return [m["text"] for m in self.messages]

    async def _tts(self, request: web.Request) -> web.StreamResponse:
        assert request.headers["Authorization"] == "Bearer k"
        if self.refuse:
            status, headers = self.refuse
            return web.json_response({"success": False, "statusCode": status}, status=status, headers=headers)
        self.connections += 1
        ws = web.WebSocketResponse(autoping=self.ping_interval is None)
        await ws.prepare(request)
        answered = asyncio.Event()
        answered.set()
        liveness = asyncio.create_task(self._drop_unanswered(ws, request, answered)) if self.ping_interval else None
        try:
            async for raw in ws:
                if raw.type == aiohttp.WSMsgType.PONG:
                    answered.set()
                elif raw.type == aiohttp.WSMsgType.PING:
                    await ws.pong(raw.data)
                elif raw.type == aiohttp.WSMsgType.TEXT and not await self._on_message(ws, json.loads(raw.data)):
                    break
        finally:
            if liveness:
                liveness.cancel()
        return ws

    async def _drop_unanswered(self, ws: web.WebSocketResponse, request: web.Request, answered: asyncio.Event) -> None:
        assert self.ping_interval and request.transport
        while True:
            await asyncio.sleep(self.ping_interval)
            if not answered.is_set():
                request.transport.close()  # no close frame: the client sees 1006, as from the real server
                return
            answered.clear()
            await ws.ping()

    async def _on_message(self, ws: web.WebSocketResponse, msg: dict) -> bool:
        if msg["type"] == "cancel" and self.answer_cancel:
            await ws.send_bytes(LATE_FRAME)
            await ws.send_str(json.dumps({"type": "cancelled", "requestId": self.running}))
        if msg["type"] != "synthesize":
            return True
        if self.unfinished:  # the server finishes the interrupted chunk before the next synthesis
            await ws.send_bytes(LATE_FRAME)
            late = {"type": self.late_event, "requestId": self.unfinished, "code": "upstream_error", "message": "x"}
            await ws.send_str(json.dumps(late))
            self.unfinished = ""
        self.running = msg["requestId"]
        if set(msg) - SYNTHESIZE_FIELDS or len(msg["text"]) > 500 or msg["voiceId"] != "v":
            await ws.send_str(json.dumps({"type": "error", "code": "validation_failed", "message": str(msg)}))
            return True
        index = len(self.messages)
        self.messages.append(msg)
        behaviour = self.plan[index] if index < len(self.plan) else speak
        if behaviour is hang_after_audio and not self.answer_cancel:
            self.unfinished = msg["requestId"]
        return await behaviour(ws, msg)

    async def _stt(self, request: web.Request) -> web.Response:
        form = await request.post()
        upload = form["file"]
        assert isinstance(upload, web.FileField)
        self.stt_forms.append(
            {
                "auth": request.headers.get("Authorization"),
                **{k: v for k, v in form.items() if k != "file"},
                "wav": upload.file.read()[:4],
            }
        )
        await asyncio.sleep(self.stt_delay)
        status, body = self.stt_response
        return web.json_response(body, status=status)

    async def _stt_stream(self, request: web.Request) -> web.StreamResponse:
        assert request.headers["Authorization"] == "Bearer k"
        if self.refuse:
            status, headers = self.refuse
            return web.json_response({"success": False, "statusCode": status}, status=status, headers=headers)
        sock = SttSocket(query=dict(request.query))
        self.stt_sockets.append(sock)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        started = {"type": "session.started", "sessionId": "s", "language": sock.query["language"]}
        await ws.send_str(json.dumps({**started, "sampleRate": 16000, "maxSessionSeconds": self.stt_max_session}))
        index = len(self.stt_sockets) - 1
        if index < len(self.stt_plan) and not await self.stt_plan[index](ws, {}):
            return ws
        async for raw in ws:
            if raw.type == aiohttp.WSMsgType.BINARY:
                assert len(raw.data) % 2 == 0 and len(raw.data) <= 64 * 1024
                sock.segments[-1] += raw.data
                heard = sum(array("h", seg).count(SPEECH) for seg in sock.segments)
                if index == 0 and self.stt_drop_after_speech is not None and heard > self.stt_drop_after_speech:
                    await ws.close(code=1011)  # mid-utterance, before its transcript
                    return ws
                continue
            if len(raw.data.encode()) > 64 * 1024:
                await ws.send_str(json.dumps({"type": "error", "code": "frame_too_large", "message": "64 KiB"}))
                await ws.close(code=1008)
                return ws
            msg = json.loads(raw.data)
            sock.order.append(msg["type"])
            if msg["type"] == "terminology":
                sock.texts.append(msg)
                await ws.send_str(json.dumps({"type": "terminology.set", "active": True}))
            elif msg["type"] == "ping":
                await ws.send_str(json.dumps({"type": "pong"}))
            elif msg["type"] in ("end_segment", "stop"):
                await self._transcribe(ws, sock)
                if msg["type"] == "stop":
                    await ws.send_str(json.dumps({"type": "session.ended", "reason": "stopped", "totalSeconds": 1}))
                    await ws.close()
        return ws

    async def _transcribe(self, ws: web.WebSocketResponse, sock: SttSocket) -> None:
        audio = sock.segments[-1]
        if not audio:
            return
        segment = len(sock.segments)
        text = f"segment {segment}" if array("h", audio).count(SPEECH) else ""
        seconds = len(audio) / 2 / STT_RATE
        event = {"type": "transcript", "segmentId": f"g{segment}", "text": text, "language": sock.query["language"]}
        await ws.send_str(json.dumps({**event, "durationSeconds": seconds, "creditsConsumed": 1, "isFinal": True}))
        sock.segments.append(b"")

    async def _voices(self, _request: web.Request) -> web.Response:
        items = [{"id": "v", "name": "ae_ahmed", "displayName": "Ahmed", "language": "ar", "gender": "Male"}]
        return web.json_response({"data": {"items": items, "total": 1}, "success": True, "statusCode": 200})

    async def __aenter__(self) -> Self:
        app = web.Application()
        app.add_routes(
            [
                web.get("/v1/text-to-speech/stream", self._tts),
                web.post("/v1/speech-to-text/sync", self._stt),
                web.get("/v1/speech-to-text/stream", self._stt_stream),
                web.get("/v1/text-to-speech/voices", self._voices),
            ]
        )
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", 0).start()
        self.url = f"http://127.0.0.1:{self._runner.addresses[0][1]}"
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._runner:
            await self._runner.cleanup()


@asynccontextmanager
async def echonova_tts(fake: FakeEchoNova, http: aiohttp.ClientSession, **kwargs: Any) -> AsyncIterator[echonova.TTS]:
    tts = echonova.TTS(api_key="k", voice_id="v", base_url=fake.url, http_session=http, **kwargs)
    try:
        yield tts
    finally:
        await tts.aclose()  # closes the pooled socket, so the fake can shut down


def ok(data: dict) -> tuple[int, dict]:
    return 200, {"data": data, "success": True, "statusCode": 200, "errorMessage": None}


def failed(status: int, message: str) -> tuple[int, dict]:
    return status, {"data": None, "success": False, "statusCode": status, "errorMessage": message}


def utterance() -> rtc.AudioFrame:
    return rtc.AudioFrame(b"\0" * 3200, 16000, 1, 1600)


def frames(value: int, seconds: float) -> list[rtc.AudioFrame]:
    """10 ms frames at 16 kHz whose every sample is `value`."""
    return [rtc.AudioFrame(array("h", [value] * 160).tobytes(), STT_RATE, 1, 160) for _ in range(round(seconds * 100))]


def streaming_stt(
    fake: FakeEchoNova, http: aiohttp.ClientSession, *, lagging: bool = False, **options: Any
) -> echonova.STT:
    opts = echonova.TranscriptionOptions(vad=cast(VAD, ScriptedVAD(lagging=lagging)), **options)
    return echonova.STT(api_key="k", base_url=fake.url, http_session=http, options=opts)


async def listen(
    stt: echonova.STT,
    audio: list[rtc.AudioFrame],
    conn: APIConnectOptions = FAST,
    *,
    flush_at: int | None = None,
    pause_at: int | None = None,
) -> list[lkstt.SpeechEvent]:
    """Push `audio`, end the input, and return every event the stream sent; `pause_at` waits 0.5 s there."""
    events: list[lkstt.SpeechEvent] = []
    async with stt.stream(conn_options=conn) as stream:

        async def push() -> None:
            for i, frame in enumerate(audio):
                stream.push_frame(frame)
                if i == flush_at:
                    stream.flush()
                await asyncio.sleep(0.5 if i == pause_at else 0.001)
            stream.end_input()

        async def drain() -> None:
            async for ev in stream:
                events.append(ev)

        results = await asyncio.wait_for(asyncio.gather(push(), drain(), return_exceptions=True), 20)
    # the stream's own error first: a push after it failed only finds the stream closed
    for result in reversed(results):
        if isinstance(result, BaseException):
            raise result
    return events


def finals(events: list[lkstt.SpeechEvent]) -> list[str]:
    return [ev.alternatives[0].text for ev in events if ev.type == SpeechEventType.FINAL_TRANSCRIPT]


async def expect(error_type: type[BaseException], coro: Awaitable) -> BaseException:
    try:
        await coro
    except error_type as e:
        return e
    raise AssertionError(f"expected {error_type.__name__}")


def spoken(audio: bytes) -> int:
    """Samples the fake sent; the SDK pads the end of a stream with silence."""
    return array("h", audio).count(1)


async def collect(stream: Any) -> tuple[bytes, list[str]]:
    audio = bytearray()
    transcripts: list[str] = []
    async for ev in stream:
        audio += ev.frame.data.tobytes()
        transcripts += [str(timed) for timed in ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])]
    return bytes(audio), transcripts


async def stream_text(tts: echonova.TTS, text: str, conn: APIConnectOptions = FAST) -> tuple[bytes, list[str]]:
    async with tts.stream(conn_options=conn) as stream:
        stream.push_text(text)
        stream.end_input()
        return await collect(stream)


def test_constructor_validation() -> None:
    os.environ.pop("ECHONOVA_API_KEY", None)
    builds: list[Callable[[], object]] = [
        echonova.STT,
        lambda: echonova.STT(api_key="k", language="fr"),
        lambda: echonova.TTS(api_key="k", voice_id=""),
        lambda: echonova.VoiceSettings(speed=3),
        lambda: echonova.VoiceSettings(cfg=0.5),
        lambda: echonova.VoiceSettings(num_steps=20),
        lambda: echonova.TranscriptionOptions(sample_rate=4000),
    ]
    for build in builds:
        try:
            build()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")

    assert echonova.STT(api_key="k").capabilities.streaming
    assert not echonova.STT(api_key="k", streaming=False).capabilities.streaming
    assert echonova.VoiceSettings(speed=1.2, tashkeel=False).payload() == {"speed": 1.2, "tashkeel": False}


async def test_stt_recognize(http: aiohttp.ClientSession) -> None:
    async with FakeEchoNova(
        stt_response=ok({"jobId": "j1", "transcription": "مرحبا", "language": "ar", "durationSeconds": 0.1})
    ) as fake:
        stt = echonova.STT(api_key="k", base_url=fake.url, http_session=http, streaming=False, language="ar-AE")
        ev = await stt.recognize(utterance(), conn_options=NO_RETRY)
        assert ev.request_id == "j1"
        assert ev.alternatives[0].text == "مرحبا" and ev.alternatives[0].language == "ar"
        assert ev.alternatives[0].end_time == 0.1

        stt.update_options(
            terminology=["KYC", {"term": "Emirates NBD", "variants": ["امريتس ان بي دي"]}], numbers_as_digits=False
        )
        await stt.recognize(utterance(), language="en-US", conn_options=NO_RETRY)
        # a language EchoNova can't take falls back to the configured one
        await stt.recognize(utterance(), language="fr", conn_options=NO_RETRY)
        # no language at all is Arabic
        unset = echonova.STT(api_key="k", base_url=fake.url, http_session=http, streaming=False, language=None)
        await unset.recognize(utterance(), conn_options=NO_RETRY)

    first, second, third, fourth = fake.stt_forms
    assert first == {"auth": "Bearer k", "language": "ar", "wav": b"RIFF"}
    assert second["language"] == "en" and second["numbers_as_digits"] == "false"
    assert json.loads(second["terminology"])["keywords"][1]["term"] == "Emirates NBD"
    assert third["language"] == "ar" and fourth["language"] == "ar"


async def test_stt_terminology_corrected(http: aiohttp.ClientSession) -> None:
    data = {"transcription": "امريتس ان بي دي", "terminology": {"corrected": "Emirates NBD"}}
    async with FakeEchoNova(stt_response=ok(data)) as fake:
        opts = echonova.TranscriptionOptions(terminology=["Emirates NBD"])
        stt = echonova.STT(api_key="k", base_url=fake.url, http_session=http, streaming=False, options=opts)
        ev = await stt.recognize(utterance(), conn_options=NO_RETRY)
    assert ev.alternatives[0].text == "Emirates NBD"


async def test_stt_errors(http: aiohttp.ClientSession) -> None:
    cases = [
        (failed(402, "Insufficient subscription balance"), False),
        (failed(429, "Monthly spending limit exhausted"), False),
        (failed(429, "ThrottlerException: Too Many Requests"), True),
        (failed(503, "Transcription is temporarily unavailable. Please try again."), True),
    ]
    for response, retryable in cases:
        async with FakeEchoNova(stt_response=response) as fake:
            stt = echonova.STT(api_key="k", base_url=fake.url, http_session=http, streaming=False)
            e = await expect(APIStatusError, stt.recognize(utterance(), conn_options=NO_RETRY))
        assert isinstance(e, APIStatusError)
        assert e.status_code == response[0] and e.retryable is retryable, e
        assert e.message == response[1]["errorMessage"]

    stt = echonova.STT(api_key="k", base_url="http://127.0.0.1:9", http_session=http, streaming=False)
    await expect(APIConnectionError, stt.recognize(utterance(), conn_options=NO_RETRY))


async def test_stt_stall_times_out(http: aiohttp.ClientSession) -> None:
    stt_module = importlib.import_module("livekit.plugins.echonova.stt")
    async with FakeEchoNova(stt_response=ok({"transcription": "late"}), stt_delay=1.0) as fake:
        stt = echonova.STT(api_key="k", base_url=fake.url, http_session=http, streaming=False)
        with mock.patch.object(stt_module, "TRANSCRIBE_TIMEOUT", 0.2):
            await expect(APITimeoutError, stt.recognize(utterance(), conn_options=NO_RETRY))


async def test_stt_stream_sends_only_speech(http: aiohttp.ClientSession) -> None:
    audio = (
        frames(SILENCE, 2.0) + frames(SPEECH, 0.6) + frames(SILENCE, 0.5) + frames(SPEECH, 0.4) + frames(SILENCE, 0.3)
    )
    async with FakeEchoNova() as fake:
        events = await listen(streaming_stt(fake, http, terminology=["KYC"]), audio)
    (sock,) = fake.stt_sockets
    assert sock.query == {"sample_rate": "16000", "language": "ar"}
    assert sock.order == ["terminology", "end_segment", "end_segment", "stop"], sock.order
    assert sock.texts[0]["terminology"] == {"keywords": ["KYC"]}
    first, second = sock.segments[:2]
    # every speech sample went out, in order, and of the 2 s of silence before it only the pre-roll did
    assert array("h", first).count(SPEECH) == 9600 and array("h", second).count(SPEECH) == 6400
    assert array("h", first).count(SILENCE) <= 16000 + 1600, len(first)
    assert finals(events) == ["segment 1", "segment 2"]
    kinds = [ev.type for ev in events if ev.type in (SpeechEventType.START_OF_SPEECH, SpeechEventType.END_OF_SPEECH)]
    assert kinds == [SpeechEventType.START_OF_SPEECH, SpeechEventType.END_OF_SPEECH] * 2
    # when speech started and stopped, so the session's endpointing delay runs from there
    edges = [ev.speech_start_time or ev.speech_end_time or 0.0 for ev in events if ev.type in kinds]
    assert all(edges) and edges == sorted(edges), edges
    spans = [(ev.alternatives[0].start_time, ev.alternatives[0].end_time) for ev in events if ev.alternatives]
    assert all(0 <= start < end for start, end in spans) and spans == sorted(spans), spans
    billed = sum(ev.recognition_usage.audio_duration for ev in events if ev.recognition_usage)
    assert abs(billed - sum(len(s) for s in sock.segments) / 2 / STT_RATE) < 1e-6


async def test_stt_stream_keeps_audio_the_vad_has_not_judged(http: aiohttp.ClientSession) -> None:
    audio = frames(SILENCE, 2.0) + frames(SPEECH, 1.5) + frames(SILENCE, 0.3)
    async with FakeEchoNova() as fake:
        events = await listen(streaming_stt(fake, http, lagging=True), audio)
    sent = array("h", b"".join(fake.stt_sockets[0].segments))
    # the VAD reached the speech only after all of it was fed: none of it is dropped
    assert sent.count(SPEECH) == 24000
    assert sent.count(SILENCE) <= 16000 + 4800, sent.count(SILENCE)  # the pre-roll and the trailing silence
    assert finals(events) == ["segment 1"]


async def test_stt_stream_terminology_fits_one_frame(http: aiohttp.ClientSession) -> None:
    # 300 Arabic terms with a variant each: inside 64 KiB as UTF-8, about three times that if escaped
    terms = [{"term": "بنك الإمارات دبي الوطني", "variants": ["امريتس ان بي دي"]}] * 300
    speech = frames(SPEECH, 0.3) + frames(SILENCE, 0.2)
    async with FakeEchoNova() as fake:
        assert finals(await listen(streaming_stt(fake, http, terminology=terms), speech)) == ["segment 1"]
    assert fake.stt_sockets[0].texts[0]["terminology"]["keywords"][0]["term"] == "بنك الإمارات دبي الوطني"

    too_many = echonova.TranscriptionOptions(terminology=terms * 3)
    try:
        echonova.STT(api_key="k", options=too_many)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a terminology list over one frame")


async def test_stt_stream_without_vad_flush_ends_segments(http: aiohttp.ClientSession) -> None:
    audio = frames(SPEECH, 0.3) + frames(SILENCE, 0.3) + frames(SPEECH, 0.3)
    async with FakeEchoNova() as fake:
        opts = echonova.TranscriptionOptions(vad=None)
        stt = echonova.STT(api_key="k", base_url=fake.url, http_session=http, options=opts, language="en-US")
        events = await listen(stt, audio, flush_at=59)
    (sock,) = fake.stt_sockets
    assert sock.query["language"] == "en"
    assert sock.order == ["end_segment", "end_segment", "stop"], sock.order
    assert sum(len(s) for s in sock.segments) == len(audio) * 320  # everything is sent
    assert finals(events) == ["segment 1", "segment 2"]


async def test_stt_stream_errors(http: aiohttp.ClientSession) -> None:
    cases: list[tuple[Behaviour, bool]] = [
        (stt_ended("insufficient_balance"), False),
        (stt_ended("upstream_error"), True),
        (close_with(4029), False),
        (close_with(1011), True),
        (error("invalid_message"), False),
    ]
    speech = frames(SPEECH, 0.3) + frames(SILENCE, 0.2)
    for behaviour, retryable in cases:
        async with FakeEchoNova(stt_plan=[behaviour]) as fake:
            e = await expect(APIError, listen(streaming_stt(fake, http), speech, NO_RETRY))
        assert isinstance(e, APIError) and e.retryable is retryable, (behaviour, e)

    for status, headers, retryable in [(400, {}, False), (429, {"Retry-After": "5"}, True), (429, {}, False)]:
        async with FakeEchoNova(refuse=(status, headers)) as fake:
            e = await expect(APIStatusError, listen(streaming_stt(fake, http), speech, NO_RETRY))
        assert isinstance(e, APIStatusError) and e.retryable is retryable and f"({status}: " in e.message, e

    # a rejected terminology list is ignored and the session goes on
    async with FakeEchoNova(stt_plan=[error("validation_failed")]) as fake:
        assert finals(await listen(streaming_stt(fake, http), speech, NO_RETRY)) == ["segment 1"]


async def test_stt_stream_reconnects(http: aiohttp.ClientSession) -> None:
    speech = frames(SILENCE, 0.2) + frames(SPEECH, 0.3) + frames(SILENCE, 0.2)
    # a dropped socket is replaced, and the call goes on on the new one
    async with FakeEchoNova(stt_plan=[close_with(1011)]) as fake:
        assert finals(await listen(streaming_stt(fake, http), speech)) == ["segment 1"]
    assert len(fake.stt_sockets) == 2

    # a socket lost mid-utterance: the next one gets that utterance whole, and nothing already transcribed
    two = speech + frames(SPEECH, 0.5) + frames(SILENCE, 0.2)
    async with FakeEchoNova(stt_drop_after_speech=4800 + 3200) as fake:
        events = await listen(streaming_stt(fake, http), two)
    first, second = fake.stt_sockets
    assert [array("h", seg).count(SPEECH) for seg in first.segments if seg] == [4800, 3200 + 160], first.segments
    assert sum(array("h", seg).count(SPEECH) for seg in second.segments) == 8000
    # the resent part is its own segment: whether speech went on during the outage is unknown
    assert finals(events) == ["segment 1", "segment 1", "segment 2"]
    kinds = [ev.type for ev in events if ev.type in (SpeechEventType.START_OF_SPEECH, SpeechEventType.END_OF_SPEECH)]
    assert kinds == [SpeechEventType.START_OF_SPEECH, SpeechEventType.END_OF_SPEECH] * (len(kinds) // 2), kinds

    # a silent socket pings, and is replaced before the session cap rather than cut by it
    stt_module = importlib.import_module("livekit.plugins.echonova.stt")
    async with FakeEchoNova(stt_max_session=1) as fake:
        with mock.patch.multiple(stt_module, KEEPALIVE_INTERVAL=0.05, SESSION_MARGIN=0.8):
            events = await listen(streaming_stt(fake, http), speech, NO_RETRY, pause_at=10)
    assert len(fake.stt_sockets) >= 2 and "ping" in fake.stt_sockets[0].order
    assert finals(events) == ["segment 1"]


async def test_tts_stream_sentences(http: aiohttp.ClientSession) -> None:
    settings = echonova.VoiceSettings(speed=1.1, cfg=3, num_steps=64, tashkeel=True, normalize=False)
    metrics: list[TTSMetrics] = []
    async with FakeEchoNova() as fake, echonova_tts(fake, http, voice_settings=settings, tokenizer=None) as tts:
        tts.on("metrics_collected", metrics.append)
        audio, transcripts = await stream_text(tts, f"{SENTENCE_A} {SENTENCE_B}")
        # the next reply reuses the pooled socket
        await stream_text(tts, SENTENCE_A)
    assert fake.synthesized == [SENTENCE_A, SENTENCE_B, SENTENCE_A], fake.synthesized
    assert fake.connections == 1
    assert spoken(audio) == 2 * SAMPLES
    assert [t.strip() for t in transcripts] == [SENTENCE_A, SENTENCE_B]
    wire = {k: fake.messages[0][k] for k in ("speed", "cfg", "numSteps", "tashkeel", "normalize")}
    assert wire == {"speed": 1.1, "cfg": 3, "numSteps": 64, "tashkeel": True, "normalize": False}
    assert [m.connection_reused for m in metrics] == [False, True], metrics


async def test_tts_stream_splits_long_sentence(http: aiohttp.ClientSession) -> None:
    text = " ".join(["word"] * 240)  # 1199 characters and no full stop
    async with FakeEchoNova() as fake, echonova_tts(fake, http) as tts:
        audio, _ = await stream_text(tts, text)
    assert all(len(t) <= 500 for t in fake.synthesized) and len(fake.synthesized) == 3
    assert " ".join(fake.synthesized) == text
    assert spoken(audio) == 3 * SAMPLES


async def test_tts_releases_audio_before_done(http: aiohttp.ClientSession) -> None:
    probe = Probe()
    heard = 0
    async with (
        FakeEchoNova(plan=[probe.burst_until_heard]) as fake,
        echonova_tts(fake, http) as tts,
        tts.stream(conn_options=FAST) as stream,
    ):
        stream.push_text(SENTENCE_A)
        stream.end_input()
        async for ev in stream:
            heard += spoken(ev.frame.data.tobytes())
            if heard >= len(FRAME) // 2:
                probe.heard.set()
    # audio held for a full emitter frame would only arrive with done: a gap on a call
    assert not probe.held


async def interrupt(tts: echonova.TTS) -> None:
    stream = tts.stream(conn_options=FAST)
    stream.push_text(SENTENCE_A)
    stream.end_input()
    await stream.__anext__()  # audio is playing
    await stream.aclose()  # barge-in


async def test_tts_interrupt_keeps_the_socket_warm(http: aiohttp.ClientSession) -> None:
    tts_module = importlib.import_module("livekit.plugins.echonova.tts")
    # however the interrupted synthesis ends (cancelled, finished, or failed), the next reply gets the
    # same warm socket at once, and none of the interrupted one's frames or events
    for answer_cancel, late_event in ((True, "done"), (False, "done"), (False, "error")):
        fake = FakeEchoNova(plan=[hang_after_audio], answer_cancel=answer_cancel, late_event=late_event)
        async with fake, echonova_tts(fake, http) as tts:
            await interrupt(tts)
            audio, _ = await stream_text(tts, SENTENCE_B)
        assert fake.connections == 1, (answer_cancel, late_event)
        assert spoken(audio) == SAMPLES and array("h", audio).count(2) == 0, (answer_cancel, late_event)

    # a reply that starts while the cancel is still being written waits for it, not for a new socket
    send_str = aiohttp.ClientWebSocketResponse.send_str

    async def slow_cancel(ws: aiohttp.ClientWebSocketResponse, data: str, compress: int | None = None) -> None:
        if '"cancel"' in data:
            await asyncio.sleep(0.2)
        await send_str(ws, data, compress)

    async with FakeEchoNova(plan=[hang_after_audio]) as fake, echonova_tts(fake, http) as tts:
        with mock.patch.object(aiohttp.ClientWebSocketResponse, "send_str", slow_cancel):
            await interrupt(tts)
            await stream_text(tts, SENTENCE_B)
    assert fake.connections == 1

    # a socket near the server's 30-minute cap is retired before it is used again
    async with FakeEchoNova() as fake, echonova_tts(fake, http) as tts:
        await stream_text(tts, SENTENCE_A)
        with mock.patch.multiple(tts_module, SESSION_SECONDS=0.05, SESSION_MARGIN=0.0):
            await asyncio.sleep(0.1)
            await stream_text(tts, SENTENCE_B)
    assert fake.connections == 2


async def test_tts_retry_rules(http: aiohttp.ClientSession) -> None:
    text = f"{SENTENCE_A} {SENTENCE_B}"
    # a later sentence that failed before any of its audio is sent again
    async with FakeEchoNova(plan=[speak, drop]) as fake, echonova_tts(fake, http) as tts:
        audio, _ = await stream_text(tts, text)
    assert fake.synthesized == [SENTENCE_A, SENTENCE_B, SENTENCE_B]
    assert spoken(audio) == 2 * SAMPLES

    cases: list[tuple[list[Behaviour], list[str]]] = [
        # part of it already played: sending it again would repeat those words
        ([speak, partial_then_drop], [SENTENCE_A, SENTENCE_B]),
        # an error a retry can't fix
        ([speak, error("voice_not_found")], [SENTENCE_A, SENTENCE_B]),
        # retries stop at max_retry
        ([speak, drop, drop, drop], [SENTENCE_A, SENTENCE_B, SENTENCE_B, SENTENCE_B]),
        # the first sentence is the framework's to retry, not ours as well
        ([drop, drop, drop], [SENTENCE_A, SENTENCE_A, SENTENCE_A]),
    ]
    for plan, sent in cases:
        async with FakeEchoNova(plan=plan) as fake, echonova_tts(fake, http) as tts:
            await expect(APIError, stream_text(tts, text))
        assert fake.synthesized == sent, (plan, fake.synthesized)


async def test_tts_errors(http: aiohttp.ClientSession) -> None:
    cases = [
        (error("voice_not_found"), False),
        (error("upstream_error"), True),
        (close_with(4002), False),
        (close_with(4008), True),
        (wrong_rate, False),
    ]
    for behaviour, retryable in cases:
        async with FakeEchoNova(plan=[behaviour]) as fake, echonova_tts(fake, http) as tts:
            e = await expect(APIError, stream_text(tts, SENTENCE_A, NO_RETRY))
        assert isinstance(e, APIError) and e.retryable is retryable, (behaviour, e)

    async with FakeEchoNova(plan=[hang_after_audio]) as fake, echonova_tts(fake, http) as tts:
        stalled = APIConnectOptions(max_retry=0, timeout=0.5)
        await expect(APITimeoutError, stream_text(tts, SENTENCE_A, stalled))

    tts = echonova.TTS(api_key="k", voice_id="v", base_url="http://127.0.0.1:9", http_session=http)
    await expect(APIConnectionError, stream_text(tts, SENTENCE_A, NO_RETRY))


async def test_tts_handshake_refusals(http: aiohttp.ClientSession) -> None:
    cases = [
        (401, {}, False),
        (402, {}, False),
        (429, {"Retry-After": "30"}, True),  # connection rate: clears after the wait
        (429, {}, False),  # spending limit or the 10-stream cap: waiting does not clear it
        (503, {}, False),  # streaming not enabled
    ]
    for status, headers, retryable in cases:
        async with FakeEchoNova(refuse=(status, headers)) as fake, echonova_tts(fake, http) as tts:
            e = await expect(APIStatusError, stream_text(tts, SENTENCE_A, NO_RETRY))
        assert isinstance(e, APIStatusError)
        assert e.status_code == status and e.retryable is retryable, e
        assert f"({status}: " in e.message, e.message


async def test_tts_synthesize_splits_long_text(http: aiohttp.ClientSession) -> None:
    text = " ".join(["word"] * 240)  # 1199 characters
    async with FakeEchoNova() as fake, echonova_tts(fake, http) as tts:
        audio = await tts.synthesize(text, conn_options=FAST).collect()
    assert [len(t) for t in fake.synthesized] == [499, 499, 199]
    assert " ".join(fake.synthesized) == text
    assert spoken(audio.data.tobytes()) == 3 * SAMPLES


async def test_list_voices(http: aiohttp.ClientSession) -> None:
    async with FakeEchoNova() as fake, echonova_tts(fake, http) as tts:
        voices = await tts.list_voices()
    assert voices[0]["id"] == "v"


async def test_tts_idle_socket_answers_server_pings(http: aiohttp.ClientSession) -> None:
    # a pooled socket nobody reads must still answer the server's pings, or the next reply finds it dropped
    async with FakeEchoNova(ping_interval=0.1) as fake, echonova_tts(fake, http) as tts:
        await stream_text(tts, SENTENCE_A)
        await asyncio.sleep(0.5)
        audio, _ = await stream_text(tts, SENTENCE_B, NO_RETRY)
    assert fake.connections == 1
    assert spoken(audio) == SAMPLES


async def main() -> None:
    test_constructor_validation()
    async with aiohttp.ClientSession() as http:
        for test in (
            test_stt_recognize,
            test_stt_terminology_corrected,
            test_stt_errors,
            test_stt_stall_times_out,
            test_stt_stream_sends_only_speech,
            test_stt_stream_keeps_audio_the_vad_has_not_judged,
            test_stt_stream_terminology_fits_one_frame,
            test_stt_stream_without_vad_flush_ends_segments,
            test_stt_stream_errors,
            test_stt_stream_reconnects,
            test_tts_stream_sentences,
            test_tts_stream_splits_long_sentence,
            test_tts_releases_audio_before_done,
            test_tts_interrupt_keeps_the_socket_warm,
            test_tts_idle_socket_answers_server_pings,
            test_tts_retry_rules,
            test_tts_errors,
            test_tts_handshake_refusals,
            test_tts_synthesize_splits_long_text,
            test_list_voices,
        ):
            await test(http)
            print("ok", test.__name__)


if __name__ == "__main__":
    asyncio.run(main())
