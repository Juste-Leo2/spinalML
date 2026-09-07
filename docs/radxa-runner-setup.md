# Radxa Self-Hosted Runner — Bootstrap Guide

Reference for initializing (or re-initializing) a Radxa Rock as a GitHub
Actions self-hosted runner. These are the exact commands applied on the board,
pin-consistent with the CI workflow (`.github/workflows/ci-simulations.yml`)
and the repo `.python-version` file.

## 1. System packages (Python 3.12 compile toolchain)

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev
```

Note: with `uv` managing the venv, most of these are only needed when
compiling CPython from source (pyenv); they are checked/idempotently installed
by the CI `prepare-env` job anyway.

## 2. pyenv (global Python 3.12.1)

```bash
curl https://pyenv.run | bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Reload the shell
source ~/.bashrc

# CRITICAL: compiling CPython must be restricted to the LITTLE cores
# (0,1,2,3) — a full big-core compile overheats/power-crashes the Radxa
# and can corrupt the SD card.
taskset -c 0,1,2,3 bash -c 'pyenv install -s 3.12.1'
pyenv global 3.12.1
pyenv rehash
```

## 3. uv + venv (from the GLOBAL interpreter)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python "$HOME/.pyenv/versions/3.12.1/bin/python" --clear
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 4. Shared gate (what the CI verifies before any test)

```bash
python3 --version               # must be 3.12.x (global, pyenv)
uv run python --version         # must be 3.12.x (active venv)
# crash reminder: cocotb VPI teardown abort ("double free or corruption",
# error -6) happens with Python <3.12 — do not downgrade below 3.12.
```

## 5. GitHub Actions runner (persistent service)

```bash
cd ~/actions-runner
./config.sh --url https://github.com/<owner>/<repo> --token <RUNNER_TOKEN>
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

## 6. Board hygiene (prevent SD corruption / power-off)

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
# /etc/systemd/logind.conf:  IdleAction=ignore / IdleActionSec=0
sudo systemctl restart systemd-logind
# Passwordless sudo for the runner user — private LAN-only CI:
sudo visudo -f /etc/sudoers.d/radxa-ci   # radxa ALL=(ALL:ALL) NOPASSWD: ALL
```
