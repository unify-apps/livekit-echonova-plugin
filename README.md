# livekit-plugins-echonova

Support for speech-to-text and text-to-speech with [EchoNova](https://echonova.ai), in Arabic dialects and
English, for [LiveKit Agents](https://github.com/livekit/agents).

## Installation

```bash
pip install "git+https://github.com/unify-apps/livekit-echonova-plugin.git@v1.0.0"
```

## Pre-requisites

You'll need an API key from EchoNova. It can be set as an environment variable: `ECHONOVA_API_KEY`.

## Usage

```python
from livekit.plugins import echonova

stt = echonova.STT(language="ar")
tts = echonova.TTS(voice_id="66e5ef3a9c1d2b4f8a7e6c01")
```

`await tts.list_voices()` lists the voices and their ids.

## Tests

```bash
pip install . && python tests/test_echonova.py
```
