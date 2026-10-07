#!/bin/bash
# Tests the menu bar menu of the real app against the real supervisor.
#
#   release/test-menu.sh SplashManager-<version>-arm64.dmg      (or a Splash Manager.app)
#
# The app is copied out of the image, started with a minimal environment and its own state, and run with
# --selftest-menu: it opens its real NSMenu in tracking mode, keeps it open for 35 s while this script starts
# Splash through the management API (so the supervisor's state really changes while the menu is open), then
# chooses the other model in the Model submenu. It fails if the menu closed or was rebuilt by itself, if the Model
# submenu was replaced, or if the choice was not saved. See Sources/SplashManager/MenuSelfTest.swift for what
# this does and does not prove (no physical mouse click).
set -uo pipefail
SRC=${1:?usage: test-menu.sh <dmg or app>}
WORK=$(mktemp -d); MGMT=8769; INFER=8203; failed=0
cleanup() {
    [ -n "${APP_PID:-}" ] && kill "$APP_PID" 2>/dev/null
    for p in $(lsof -nP -iTCP:$INFER -sTCP:LISTEN -t 2>/dev/null); do kill -TERM "$p" 2>/dev/null; done
    security delete-generic-password -s net.2elemental.splash-manager.menutest -a management-token >/dev/null 2>&1
    hdiutil detach "$WORK/mnt" -quiet >/dev/null 2>&1
    rm -rf "$WORK"
}
trap cleanup EXIT
lsof -nP -iTCP:$MGMT -iTCP:$INFER -sTCP:LISTEN >/dev/null 2>&1 && { echo "ports $MGMT/$INFER are in use"; exit 2; }
mkdir -p "$WORK/mnt" "$WORK/state/support"
if [ -d "$SRC" ]; then ditto "$SRC" "$WORK/Splash Manager.app"
else hdiutil attach "$SRC" -nobrowse -readonly -mountpoint "$WORK/mnt" -quiet && ditto "$WORK/mnt/Splash Manager.app" "$WORK/Splash Manager.app" && hdiutil detach "$WORK/mnt" -quiet; fi
EXE="$WORK/Splash Manager.app/Contents/MacOS/SplashManager"

python3 - "$WORK/state/support/config.json" "$MGMT" "$INFER" <<'PY' || { echo "no Splash-pinned model in the Hugging Face cache"; exit 2; }
import glob, json, os, sys
out, mgmt, infer = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cache = os.environ.get("HF_HUB_CACHE") or (os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else os.path.expanduser("~/.cache/huggingface/hub"))
cands = [(os.path.basename(r)[8:].replace("--", "/"), os.path.basename(p)) for r in sorted(glob.glob(os.path.join(cache, "models--*")))
         if not r.lower().endswith("dflash2") for p in glob.glob(os.path.join(r, "refs", "splash", "*", "*"))]
if not cands: sys.exit(1)
model, rev = cands[0]
cfg = lambda i, n, **o: {"id": i, "displayName": n, "modelId": model, "revision": rev, "options": {**o}}
json.dump({"configs": [cfg("base", "Base"), cfg("ctx64k", "64K context", maxContext="64K")],
           "settings": {"inferencePort": infer, "managementPort": mgmt, "selectedConfigId": "base"}}, open(out, "w"))
PY

ENVV=(env -i HOME="$HOME" PATH=/usr/bin:/bin SPLASH_MANAGER_HOME="$WORK/state" SPLASH_MANAGER_KEYCHAIN_SERVICE=net.2elemental.splash-manager.menutest)
TOKEN=$("${ENVV[@]}" "$EXE" --print-token)
( cd / && exec "${ENVV[@]}" "$EXE" --selftest-menu --selftest-report "$WORK/report.json" --selftest-hold ${HOLD:-35} > "$WORK/app.out" 2>&1 ) & APP_PID=$!
B=http://127.0.0.1:$MGMT/api/v1
for _ in $(seq 1 40); do curl -s -m 1 -o /dev/null -H "Authorization: Bearer $TOKEN" $B/status && break; sleep 0.5; done
sleep 4   # the menu opens 2 s after launch; from here on it is open
echo "menu is open; starting Splash through the API so the state changes under it"
curl -s -m 170 -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"config_id":"base","wait_ready_seconds":120}' -X POST $B/start -o /dev/null
wait $APP_PID; rc=$?; APP_PID=
echo "self test exit code $rc"
[ -f "$WORK/report.json" ] && cat "$WORK/report.json" || { echo "no report"; cat "$WORK/app.out" | head; exit 1; }
exit $rc
