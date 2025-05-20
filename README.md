# MarbleNet inference

[![CI pipeline](https://github.com/egorsmkv/marblenet-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/egorsmkv/marblenet-inference/actions/workflows/ci.yml)

This repo contains workable code to run NeMo's model that does Voice Activity Detection.

## Install

```bash
uv venv --python 3.12

source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

## Testing

See the VAD pieces in a file:

```bash
python inference.py
```

## Misc

See the VAD pieces in a file:

```bash
python view_rttm.py rttm_outputs/audio_file.rttm
```
