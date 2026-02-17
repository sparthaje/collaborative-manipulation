# Collaborative Manipulation

This repo is set up as a `uv` project for the SO-100 workflow from:
`https://deepwiki.com/box2ai-robotics/lerobot-joycon/7.1-getting-started-with-so-100`

## Prerequisites

- Linux: `sudo apt-get install -y ffmpeg libsm6 libxext6`
- Python 3.10 (managed by `uv`)

## Setup with uv

```bash
# create/sync the local .venv using Python 3.10
uv sync --python 3.10

# include optional JoyCon-related packages
uv sync --python 3.10 --group joycon
```

## Common commands

```bash
# verify install
uv run python -c "import lerobot; print('lerobot ok')"

# update all dependencies to latest allowed versions and refresh lockfile
uv lock --upgrade
uv sync
```

## SO-100 calibration/teleop (examples)

Use `uv run` in front of the same Python commands from the upstream docs, for example:

```bash
uv run python -m lerobot.calibrate --help
uv run python -m lerobot.teleoperate --help
uv run python -m lerobot.record --help
```

If your hardware setup needs additional system packages or udev rules, keep following the upstream hardware section from the DeepWiki page.
