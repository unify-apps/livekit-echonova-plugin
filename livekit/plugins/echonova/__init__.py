from . import models
from .stt import STT, SpeechStream, TranscriptionOptions
from .tts import TTS, ChunkedStream, SynthesizeStream, VoiceSettings
from .version import __version__

__all__ = [
    "STT",
    "TTS",
    "ChunkedStream",
    "SpeechStream",
    "SynthesizeStream",
    "TranscriptionOptions",
    "VoiceSettings",
    "__version__",
    "models",
]


from livekit.agents import Plugin

from .log import logger


class EchoNovaPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__ or __name__, logger)


Plugin.register_plugin(EchoNovaPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
