#!/usr/bin/env bash
#
# run.sh — start Specter LLM
#
# Checks the environment, installs anything missing, makes sure Ollama is up
# with the models this project needs, then launches Streamlit and opens it in
# your browser.
#
# Usage:
#   ./run.sh                 normal start
#   ./run.sh --port 8502     use a different port
#   ./run.sh --no-browser    start without opening a browser
#   ./run.sh --reinstall     force dependency reinstall
#   ./run.sh --skip-checks   skip Ollama/dependency checks (fastest start)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR=".venv"
PORT=8501
OPEN_BROWSER=1
FORCE_REINSTALL=0
SKIP_CHECKS=0
REQUIRED_MODELS=("llama3.2" "nomic-embed-text")
STREAMLIT_PID=""

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

if [ -t 1 ]; then
    RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'
    BLUE=$'\033[0;34m'; BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    RED=""; GREEN=""; YELLOW=""; BLUE=""; BOLD=""; RESET=""
fi

info()  { printf "%s==>%s %s\n" "$BLUE" "$RESET" "$1"; }
ok()    { printf "%s  ok%s %s\n" "$GREEN" "$RESET" "$1"; }
warn()  { printf "%s  !!%s %s\n" "$YELLOW" "$RESET" "$1"; }
die()   { printf "%s error:%s %s\n" "$RED" "$RESET" "$1" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)        PORT="${2:-}"; [ -n "$PORT" ] || die "--port needs a value"; shift 2 ;;
        --no-browser)  OPEN_BROWSER=0; shift ;;
        --reinstall)   FORCE_REINSTALL=1; shift ;;
        --skip-checks) SKIP_CHECKS=1; shift ;;
        # Print the header comment block as the help text, stopping at the
        # first line of actual code so it cannot drift out of sync.
        -h|--help)     awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
        *)             die "unknown option: $1  (try --help)" ;;
    esac
done

# Shut Streamlit down cleanly on Ctrl+C rather than orphaning it.
cleanup() {
    if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        printf "\n"; info "Stopping Specter..."
        kill "$STREAMLIT_PID" 2>/dev/null || true
        wait "$STREAMLIT_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

printf "\n%s┌───────────────────────────────────────┐%s\n" "$BOLD" "$RESET"
printf "%s│   Specter LLM — contract analyser     │%s\n" "$BOLD" "$RESET"
printf "%s└───────────────────────────────────────┘%s\n\n" "$BOLD" "$RESET"

# ---------------------------------------------------------------------------
# 1. Python
# ---------------------------------------------------------------------------

info "Checking Python..."

command -v python3 >/dev/null 2>&1 || die "python3 not found. Install Python 3.11+ from https://python.org"

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_OK="$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 11) else 0)')"
[ "$PY_OK" = "1" ] || die "Python 3.11+ required (this project pins >=3.11), found $PY_VERSION"
ok "Python $PY_VERSION"

# uv is much faster and the project ships a uv.lock, so prefer it when present.
if command -v uv >/dev/null 2>&1; then
    USE_UV=1; ok "uv found — using it for dependency management"
else
    USE_UV=0; warn "uv not found — falling back to venv + pip (slower)"
fi

# ---------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------

info "Checking virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    warn "No $VENV_DIR found — creating one"
    if [ "$USE_UV" = "1" ]; then
        uv venv "$VENV_DIR" || die "failed to create virtual environment with uv"
    else
        python3 -m venv "$VENV_DIR" || die "failed to create virtual environment"
    fi
    ok "Created $VENV_DIR"
    FORCE_REINSTALL=1
else
    ok "Found $VENV_DIR"
fi

VENV_PY="$PROJECT_DIR/$VENV_DIR/bin/python"
[ -x "$VENV_PY" ] || die "$VENV_PY is missing or not executable. Delete $VENV_DIR and re-run."

# ---------------------------------------------------------------------------
# 3. Dependencies
# ---------------------------------------------------------------------------

if [ "$SKIP_CHECKS" = "1" ] && [ "$FORCE_REINSTALL" = "0" ]; then
    warn "Skipping dependency check (--skip-checks)"
else
    info "Checking dependencies..."

    # Import-check the packages the app actually needs at startup. Cheaper and
    # more honest than trusting a requirements file to match what is installed.
    NEEDS_INSTALL=0
    if [ "$FORCE_REINSTALL" = "1" ]; then
        NEEDS_INSTALL=1
    elif ! "$VENV_PY" -c "import streamlit, fitz, chromadb, ollama, networkx, requests" >/dev/null 2>&1; then
        warn "Some required packages are missing"
        NEEDS_INSTALL=1
    fi

    if [ "$NEEDS_INSTALL" = "1" ]; then
        info "Installing dependencies (first run takes a few minutes)..."
        if [ "$USE_UV" = "1" ] && [ -f "uv.lock" ]; then
            uv sync || die "uv sync failed"
        elif [ "$USE_UV" = "1" ] && [ -f "requirements.txt" ]; then
            uv pip install --python "$VENV_PY" -r requirements.txt || die "uv pip install failed"
        elif [ -f "requirements.txt" ]; then
            "$VENV_PY" -m pip install --upgrade pip >/dev/null 2>&1 || true
            "$VENV_PY" -m pip install -r requirements.txt || die "pip install failed"
        else
            die "no uv.lock or requirements.txt found — cannot install dependencies"
        fi
        ok "Dependencies installed"
    else
        ok "All required packages present"
    fi
fi

# ---------------------------------------------------------------------------
# 4. Ollama
# ---------------------------------------------------------------------------
# Specter runs entirely locally: llama3.2 answers questions, nomic-embed-text
# builds the embeddings. Without both, the app loads but every query fails.

if [ "$SKIP_CHECKS" = "1" ]; then
    warn "Skipping Ollama check (--skip-checks)"
else
    info "Checking Ollama..."

    if ! command -v ollama >/dev/null 2>&1; then
        warn "Ollama is not installed — install it from https://ollama.com"
        warn "Specter will start, but every question and ingest will fail without it."
    else
        ok "Ollama installed"

        # Is the server actually up? The CLI existing does not mean it is serving.
        if ! curl -fsS --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
            warn "Ollama server not responding — starting it in the background"
            nohup ollama serve >/dev/null 2>&1 &
            for _ in $(seq 1 20); do
                sleep 0.5
                curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1 && break
            done
        fi

        if curl -fsS --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
            ok "Ollama server running on :11434"

            INSTALLED_MODELS="$(ollama list 2>/dev/null || true)"
            for model in "${REQUIRED_MODELS[@]}"; do
                if grep -q "^${model}[: ]" <<<"$INSTALLED_MODELS"; then
                    ok "Model '$model' ready"
                else
                    warn "Model '$model' missing — pulling it now (this can take a while)"
                    ollama pull "$model" || warn "could not pull '$model' — the app will fail on queries needing it"
                fi
            done
        else
            warn "Ollama server did not come up. Start it yourself with: ollama serve"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 5. Port
# ---------------------------------------------------------------------------

info "Checking port $PORT..."

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    warn "Port $PORT is already in use — trying the next free port"
    for candidate in $(seq $((PORT + 1)) $((PORT + 20))); do
        if ! lsof -nP -iTCP:"$candidate" -sTCP:LISTEN >/dev/null 2>&1; then
            PORT="$candidate"; break
        fi
    done
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && die "no free port found near $PORT"
fi
ok "Using port $PORT"

# ---------------------------------------------------------------------------
# 6. Launch
# ---------------------------------------------------------------------------

[ -f "app.py" ] || die "app.py not found in $PROJECT_DIR"

URL="http://localhost:$PORT"
printf "\n"
info "Starting Specter on $BOLD$URL$RESET"
printf "    Press %sCtrl+C%s to stop.\n\n" "$BOLD" "$RESET"

# Streamlit's own browser-opening is disabled so we can wait until the server
# is genuinely accepting connections first — otherwise the tab races the boot
# and lands on a connection error.
"$VENV_PY" -m streamlit run app.py \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false &
STREAMLIT_PID=$!

SERVER_UP=0
for _ in $(seq 1 60); do
    sleep 0.5
    if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        die "Streamlit exited during startup — see the output above"
    fi
    if curl -fsS --max-time 2 "$URL" >/dev/null 2>&1; then
        SERVER_UP=1; break
    fi
done

if [ "$SERVER_UP" = "1" ]; then
    ok "Specter is up"
    if [ "$OPEN_BROWSER" = "1" ]; then
        case "$(uname -s)" in
            Darwin) open "$URL" >/dev/null 2>&1 || warn "could not open browser — visit $URL" ;;
            Linux)  xdg-open "$URL" >/dev/null 2>&1 || warn "could not open browser — visit $URL" ;;
            *)      warn "unrecognised platform — open $URL yourself" ;;
        esac
    fi
else
    warn "Server did not respond in 30s — it may still be starting. Try $URL"
fi

# Hand the terminal back to Streamlit so its logs stream and Ctrl+C reaches us.
wait "$STREAMLIT_PID"
