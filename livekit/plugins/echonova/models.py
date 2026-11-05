from typing import Literal

# https://echonova.ai/docs/reference/stt-transcribe-sync — `language`; "ar" also covers mixed Arabic/English.
STTLanguages = Literal["ar", "en"]

# https://echonova.ai/docs/reference/tts-stream — `numSteps`.
TTSNumSteps = Literal[16, 32, 48, 64, 128]
