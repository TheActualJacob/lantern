#!/usr/bin/env bash
# Pull the latest (or a named) Lantern Recorder session off the device into ./pulled_sessions/.
#
# Usage:
#   ./pull_session.sh                 # pull the most-recent session
#   ./pull_session.sh session_2026..  # pull a specific session by name
#   ./pull_session.sh --all           # pull every session on the device
#
# Sessions live at /sdcard/Android/data/com.lantern.recorder/files/sessions/ on the device.
set -euo pipefail

ADB="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
command -v "$ADB" >/dev/null 2>&1 || ADB="adb"

REMOTE_ROOT="/sdcard/Android/data/com.lantern.recorder/files/sessions"
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)/pulled_sessions"
mkdir -p "$LOCAL_ROOT"

list_remote() { "$ADB" shell ls -t "$REMOTE_ROOT" 2>/dev/null | tr -d '\r' | grep -v '^$'; }

pull_one() {
  local name="$1"
  echo ">> pulling $name"
  "$ADB" pull "$REMOTE_ROOT/$name" "$LOCAL_ROOT/" >/dev/null
  local n
  n=$(ls "$LOCAL_ROOT/$name"/frame_*.json 2>/dev/null | wc -l | tr -d ' ')
  echo "   -> $LOCAL_ROOT/$name ($n frames)"
}

case "${1:-}" in
  --all)
    for s in $(list_remote); do pull_one "$s"; done ;;
  "")
    latest="$(list_remote | head -n1)"
    [ -n "$latest" ] || { echo "No sessions found on device."; exit 1; }
    pull_one "$latest" ;;
  *)
    pull_one "$1" ;;
esac
