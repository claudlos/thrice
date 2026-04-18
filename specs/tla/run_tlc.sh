#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_tlc.sh - run the TLC model checker on the Thrice TLA+ specs.
#
# Usage:
#     ./specs/tla/run_tlc.sh               # check both specs (default bounds)
#     ./specs/tla/run_tlc.sh cron          # only MCCronJob
#     ./specs/tla/run_tlc.sh agent         # only MCAgentLoop
#     TLC_WORKERS=auto ./specs/tla/run_tlc.sh
#
# Requires: Java 11+ on PATH.  The script auto-downloads tla2tools.jar
# the first time you run it (released by Microsoft/tlaplus on GitHub).
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

TLA_JAR_URL="${TLA_JAR_URL:-https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar}"
TLA_JAR="${TLA_JAR:-${SCRIPT_DIR}/tla2tools.jar}"
TLC_WORKERS="${TLC_WORKERS:-auto}"
TLC_DEADLOCK_FLAG=""   # deadlock is disabled per model in the .cfg files
TLC_EXTRA_ARGS="${TLC_EXTRA_ARGS:-}"

WHICH="${1:-all}"

# ---- Dependencies ------------------------------------------------------------
if ! command -v java >/dev/null 2>&1; then
    echo "ERROR: 'java' not found on PATH. Install a JRE/JDK 11+ and retry." >&2
    exit 2
fi

if [ ! -f "$TLA_JAR" ]; then
    echo ">>> tla2tools.jar not found, downloading from:"
    echo "    $TLA_JAR_URL"
    if command -v curl >/dev/null 2>&1; then
        curl -fL -o "$TLA_JAR" "$TLA_JAR_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$TLA_JAR" "$TLA_JAR_URL"
    else
        echo "ERROR: need curl or wget to download tla2tools.jar" >&2
        exit 2
    fi
fi

# ---- Helper ------------------------------------------------------------------
run_tlc() {
    local module="$1"
    local cfg="$2"
    echo
    echo "================================================================"
    echo "  TLC: $module  (config=$cfg, workers=$TLC_WORKERS)"
    echo "================================================================"
    # -workers auto lets TLC pick a reasonable thread count
    # -deadlock disabled by the .cfg
    # -cleanup removes the states/ dir from previous runs
    java -XX:+UseParallelGC -jar "$TLA_JAR" \
        -workers "$TLC_WORKERS" \
        -config "$cfg" \
        -cleanup \
        $TLC_EXTRA_ARGS \
        "$module"
}

# ---- Targets ----------------------------------------------------------------
status=0
case "$WHICH" in
    cron|CronJob)
        run_tlc MCCronJob MCCronJob.cfg || status=$?
        ;;
    agent|AgentLoop)
        run_tlc MCAgentLoop MCAgentLoop.cfg || status=$?
        ;;
    bisector|Bisector)
        run_tlc MCBisector MCBisector.cfg || status=$?
        ;;
    all|*)
        run_tlc MCCronJob   MCCronJob.cfg    || status=$?
        run_tlc MCAgentLoop MCAgentLoop.cfg  || status=$?
        run_tlc MCBisector  MCBisector.cfg   || status=$?
        ;;
esac

exit $status
