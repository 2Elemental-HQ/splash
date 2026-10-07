#!/bin/bash
# End-to-end test of a built disk image on a Mac that has the model in its Hugging Face cache.
# It uses nothing from the source checkout: the app is copied out of the image into a temporary folder,
# the image is unmounted, the app runs with a minimal environment and its own state, and everything goes
# through the app's management API and Splash's HTTP endpoint.
#
#   release/test-package.sh SplashManager-<version>-arm64.dmg
#
# Checks: runtime found, verified, can drain; start and readiness; a short real inference call; drain during a
# running stream (the stream finishes, a new call is refused, a second operation is refused); switch to a
# configuration with other options; stop. Downloads nothing: it needs the cached model (offline start).
set -uo pipefail
DMG=${1:?usage: test-package.sh <dmg>}
WORK=$(mktemp -d); MNT="$WORK/mnt"; APPS="$WORK/Applications"; ISO="$WORK/state"
MGMT=8767; INFER=8201
pass=0; failed=0
ok()   { echo "  ok    $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $1"; failed=$((failed+1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; echo "        last status: $(echo "${S:-}" | head -c 600)"; fi; }
cleanup() {
    [ -n "${APP_PID:-}" ] && kill "$APP_PID" 2>/dev/null && sleep 1
    for p in $(lsof -nP -iTCP:$INFER -sTCP:LISTEN -t 2>/dev/null); do kill -TERM "$p" 2>/dev/null; done
    security delete-generic-password -s net.2elemental.splash-manager.packagetest -a management-token >/dev/null 2>&1
    hdiutil detach "$MNT" -quiet >/dev/null 2>&1
    rm -rf "$WORK"
}
trap cleanup EXIT
lsof -nP -iTCP:$MGMT -iTCP:$INFER -sTCP:LISTEN >/dev/null 2>&1 && { echo "ports $MGMT/$INFER are in use"; exit 2; }

mkdir -p "$MNT" "$APPS" "$ISO/support"
hdiutil attach "$DMG" -nobrowse -readonly -mountpoint "$MNT" -quiet || { echo "cannot mount $DMG"; exit 2; }
ditto "$MNT/Splash Manager.app" "$APPS/Splash Manager.app"
hdiutil detach "$MNT" -quiet
APP="$APPS/Splash Manager.app"; EXE="$APP/Contents/MacOS/SplashManager"
echo "Package test of $(basename "$DMG") from $APPS (checkout not used)"

# Seed two configurations from the model Splash pinned in the Hugging Face cache; no model id is typed here.
python3 - "$ISO/support/config.json" "$MGMT" "$INFER" <<'PY' || { echo "no Splash-pinned model in the Hugging Face cache"; exit 2; }
import glob, json, os, sys
out, mgmt, infer = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cache = os.environ.get("HF_HUB_CACHE") or (os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else os.path.expanduser("~/.cache/huggingface/hub"))
cands = []
for repo in sorted(glob.glob(os.path.join(cache, "models--*"))):
    name = os.path.basename(repo)[len("models--"):].replace("--", "/")
    if name.lower().endswith("dflash2"): continue
    for pin in glob.glob(os.path.join(repo, "refs", "splash", "*", "*")):
        cands.append((name, os.path.basename(pin)))
if not cands: sys.exit(1)
model, rev = cands[0]
def cfg(i, n, **o): return {"id": i, "displayName": n, "modelId": model, "revision": rev, "options": {"disableANE": False, "languageOnly": False, "reasoning": "model_default", **o}}
json.dump({"configs": [cfg("base", "Base"), cfg("ctx64k", "64K context", maxContext="64K")],
           "settings": {"inferencePort": infer, "managementPort": mgmt, "selectedConfigId": "base", "allowedHosts": [],
                        "autoRestart": True, "inferenceExposure": "loopback", "managementExposure": "loopback",
                        "managementEnabled": True, "persistLogsToFile": False, "startSplashWhenAppLaunches": False,
                        "stopSplashWhenAppQuits": True, "preferInstalledSplash": False}}, open(out, "w"))
print(model, rev[:8])
PY

ENVV=(env -i HOME="$HOME" PATH=/usr/bin:/bin SPLASH_MANAGER_HOME="$ISO" SPLASH_MANAGER_KEYCHAIN_SERVICE=net.2elemental.splash-manager.packagetest)
"${ENVV[@]}" "$EXE" --verify-runtime && ok "runtime integrity (the app's own check)" || bad "runtime integrity"
TOKEN=$("${ENVV[@]}" "$EXE" --print-token)
( cd / && exec "${ENVV[@]}" "$EXE" > "$WORK/app.out" 2>&1 ) & APP_PID=$!
B=http://127.0.0.1:$MGMT/api/v1
for _ in $(seq 1 30); do curl -s -m 1 -o /dev/null -H "Authorization: Bearer $TOKEN" $B/status && break; sleep 0.5; done
api() { curl -s -m "${3:-130}" -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" ${2:+-d "$2"} -X "${4:-GET}" "$B$1"; }
jq_() { python3 -c "import json,sys; d=json.load(sys.stdin)
try: print(eval(sys.argv[1]))
except Exception as e: print('ERR', e)" "$1"; }

S=$(api /status)
echo "  start-up status: $(echo "$S" | jq_ 'd["splash"]')"
check "bundled runtime is used"            '[ "$(echo "$S" | jq_ "d[\"splash\"][\"source\"]")" = bundled ]'
check "runtime integrity is reported"      '[ "$(echo "$S" | jq_ "d[\"splash\"][\"integrity\"]")" = verified ]'
check "runtime version is the drain build" '[ "$(echo "$S" | jq_ "d[\"splash\"][\"version\"]")" = 1.3.0-drain.1 ]'
check "configurations came from the cache" '[ "$(api /configs | jq_ "len(d[\"configs\"])")" = 2 ]'

api /start '{"config_id":"base","wait_ready_seconds":150}' 170 POST > "$WORK/start.json"
S=$(api /status)
check "started and ready (model listed, HTTP ready)" '[ "$(echo "$S" | jq_ "d[\"state\"]+str(d[\"readiness\"][\"model_loaded\"])")" = readyTrue ]'
check "the running Splash reports drain support"   '[ "$(echo "$S" | jq_ "d[\"drain\"][\"supported\"]")" = True ]'
check "switch_safe is true only because of drain"  '[ "$(echo "$S" | jq_ "d[\"activity\"][\"switch_safe\"]")" = True ]'
MODEL=$(echo "$S" | jq_ 'd["applied"]["model_id"]')

R=$(curl -s -m 60 localhost:$INFER/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":50,\"reasoning_effort\":\"none\"}")
check "a real short inference call answers" '[ "$(echo "$R" | jq_ "d[\"choices\"][0][\"message\"][\"content\"].strip()")" = pong ]'

# A long stream, then a switch to the other configuration while it runs.
curl -s -N -m 120 localhost:$INFER/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write 600 words about Antwerp harbour.\"}],\"max_tokens\":700,\"stream\":true,\"reasoning_effort\":\"none\"}" > "$WORK/stream.out" &
STREAM=$!
sleep 3
SW=$(api /switch '{"config_id":"ctx64k"}' 30 POST)
check "switch answers at once with state stopping (drain started)" '[ "$(echo "$SW" | jq_ "d[\"state\"]")" = stopping ]'
sleep 1
NEW=$(curl -s -m 10 -w ' %{http_code}' localhost:$INFER/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}")
check "a new call during the drain is refused (503 server_draining)" 'echo "$NEW" | grep -q server_draining && echo "$NEW" | grep -q " 503$"'
check "a second stop during the switch is refused (busy)" '[ "$(api /stop "" 20 POST | jq_ "d[\"error\"][\"code\"]")" = busy ]'
wait $STREAM
check "the stream that was running finished normally (length)" 'grep -q "\"finish_reason\":\"length\"" "$WORK/stream.out" && grep -q "\[DONE\]" "$WORK/stream.out" && ! grep -q server_shutdown "$WORK/stream.out"'
S=$(api "/status?wait_for=ready&timeout=150" "" 170)
check "the other configuration runs after the drain" '[ "$(echo "$S" | jq_ "d[\"state\"]+d[\"applied\"][\"config_id\"]")" = readyctx64k ]'
check "its option reached Splash (context 65536)" '[ "$(curl -s localhost:$INFER/status | jq_ "d[\"maximum_context_tokens\"]")" = 65536 ]'
R=$(curl -s -m 60 localhost:$INFER/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":50,\"reasoning_effort\":\"none\"}")
check "inference works again after the switch" '[ "$(echo "$R" | jq_ "d[\"choices\"][0][\"message\"][\"content\"].strip()")" = pong ]'

S=$(api /stop '{"wait_stopped_seconds":60}' 90 POST)
check "stop drains and ends in state stopped" '[ "$(echo "$S" | jq_ "d[\"state\"]")" = stopped ]'
check "no Splash process is left" '! pgrep -f "Splash Manager.app/Contents/Resources/Splash" >/dev/null'
echo; echo "$pass passed, $failed failed"; [ "$failed" = 0 ]
