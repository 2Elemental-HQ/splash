# Management API, version 1

Handover contract for AgentOS (PR #106 and later). This API manages the Splash
process. It does not carry inference: AgentOS keeps calling Splash's own
OpenAI-compatible endpoint (`<endpoint>/v1`).

## Where it runs and how to reach it

The API runs inside the Splash Manager app, not inside Splash. It answers while
Splash is stopped, crashed or loading, and while the management window is
closed. It stops only when the app quits. Turn on **Settings → Open at login**
so the app is always there after a reboot.

| Setting | Listens on | Use |
| --- | --- | --- |
| Management: *This Mac only* (default) | `127.0.0.1:8765` | Local tools. Not reachable from the tailnet. |
| Management: *Tailscale address and this Mac* | `127.0.0.1:8765` and `<tailscale-ip>:8765` | AgentOS on the VPS. Never `0.0.0.0`. |

The port is configurable. The transport is plain HTTP; on a tailnet the traffic
is already encrypted by WireGuard. To get HTTPS, keep *This Mac only* and publish
the loopback port with `tailscale serve`.

A Mac cannot connect to its own Tailscale address, so test the tailnet path
from another device, for example from the VPS:

```sh
curl -sS -H "Authorization: Bearer $SPLASH_MANAGER_TOKEN" http://<mac-tailnet-name-or-ip>:8765/api/v1/status
```

## Authentication

Every route needs `Authorization: Bearer <token>`. The token is 256 random bits,
created at first run, stored in the macOS Keychain (service
`net.2elemental.splash-manager`). Copy it from **Settings → Management API**, or
run `"Splash Manager.app/Contents/MacOS/SplashManager" --print-token`. A wrong
token gets 401 after a 300 ms delay. The API never returns the token or the
inference API key.

## Conventions

* JSON in, JSON out, `snake_case`, ISO 8601 times.
* POST bodies accept the listed fields only. Any other field gets 400
  `invalid_request`. There is no field for a command, an argument or a model id:
  `config_id` names a configuration that a person saved in the app.
* Errors: `{"error": {"code", "message", "details"?}}`.
* Bodies are limited to 16 KiB. One request per connection.

## Status object

`GET /api/v1/status` (and the body of start, switch and stop):

```json
{
  "api_version": 1,
  "manager": {"version": "0.1.0", "started_at": "…"},
  "state": "stopped | starting | ready | stopping | failed",
  "ownership": "none | managed | external",
  "detail": "human text or null",
  "splash": {"installed": true, "version": "1.3.0"},
  "config": {"id": "…", "display_name": "…", "model_id": "…", "revision": "…|null"},
  "loaded_model_id": "model id that /v1/models lists, or null",
  "endpoint": {
    "bind_host": "127.0.0.1 | 0.0.0.0", "port": 8000,
    "exposure": "loopback | all_interfaces", "requires_api_key": false,
    "local_url": "http://127.0.0.1:8000", "tailnet_url": "http://100.x.y.z:8000 | null",
    "openai_base_path": "/v1", "allowed_hosts": []
  },
  "readiness": {"process_alive": true, "http_ready": true, "model_loaded": true},
  "activity": {"active_requests": 0, "idle": true, "switch_safe": true, "reason": null},
  "process": {"pid": 1234, "started_at": "…", "uptime_seconds": 60, "adopted": false},
  "last_error": {"code": "…", "message": "…", "at": "…"},
  "retry": {"attempt": 1, "max_attempts": 3, "next_at": "…"},
  "conflict": {"kind": "external_splash | foreign_service | unreadable_service", "pid": 1, "model_id": "…", "message": "…"}
}
```

* `state: "ready"` means the process runs, `/ready` answered 200 **and**
  `/v1/models` lists the intended model. A process that merely exists is `starting`.
* `ownership: "managed"`: the app started this process (or re-adopted it after an
  app restart, after checking pid, kernel start time and Splash's own
  `/status.instance.pid`). `"external"`: another Splash already serves on the
  port. The app watches it and never stops it.
* `activity.switch_safe` is true only when the last reading proved that no
  request is running or waiting.
* `retry` is set while a restart after a crash is scheduled.
* `endpoint.requires_api_key` is true when Splash is exposed on all interfaces.
  The inference key is held in the Keychain (`inference-api-key`) and shown in the
  app. It is not part of this API; configure it once in AgentOS.

## Routes

### `GET /api/v1/status`

Optional `?wait_for=ready|stopped|…&timeout=30` (seconds, max 600) holds the
response until the state is reached, a start failed for good, or the timeout
ends. Always returns 200 with the current status; compare `state`.

### `GET /api/v1/configs`

```json
{"configs": [{
  "id": "qwen3-8-27b-4bit-10c35ca", "display_name": "…", "model_id": "…", "revision": "…|null",
  "availability": "loaded | local | not_local",
  "selected": true, "active": false,
  "download_required": false, "missing": [], "note": null,
  "options": {"language_only": false, "max_context": null, "max_memory": null,
              "idle_release": null, "reasoning_default": "model_default | off", "disable_ane": false}
}]}
```

`configured` is every row. `local` means the files are in the Hugging Face cache
(Splash then starts with `--offline`). `loaded` means this configuration is the
model that serves now. `missing` lists repositories a start would download.

### `GET /api/v1/configs/{id}/download-estimate`

Asks the Hugging Face Hub for file sizes (the only call that leaves the Mac).
`{"config_id", "download_required", "missing": [], "estimates": [{"repository", "bytes", "error"?}]}`.

### `POST /api/v1/start`

`{"config_id": "…", "allow_download": false, "wait_ready_seconds": 0}`

* Idempotent: a repeat for the same configuration returns the current status.
  Concurrent calls start one process.
* 200 when `state` is `ready`, otherwise 202 (still starting). Set
  `wait_ready_seconds` (0–600) to hold the call until ready, failed or timeout.
* 409 `download_required` unless `allow_download` is true and the files are missing.
  Call the estimate route, show the size, then repeat with `allow_download: true`.

### `POST /api/v1/switch`

Same body as start. Loads another configuration: it checks that no call runs,
stops the current process gracefully, then starts the other one. If the
target is already active it returns the status. If nothing runs it acts as start.

### `POST /api/v1/stop`

Empty body. Stops the managed process (SIGTERM, then SIGINT after 40 s, then
SIGKILL after 10 s more). There is no `force` over the API.

### `GET /api/v1/logs?lines=100`

The last lines (max 500) of app and Splash console output. Redacted on entry.
Splash 1.3.0 prints metadata only (timings, token counts); prompts and answers
are not in it.

## Error codes

| HTTP | `code` | Meaning |
| --- | --- | --- |
| 400 | `invalid_request` | Unknown field, bad type, bad JSON. |
| 401 | `unauthorized` | Missing or wrong token. |
| 404 | `unknown_config`, `not_found` | No such configuration or route. |
| 405 | `method_not_allowed` | |
| 409 | `active_requests` | Calls run or wait; nothing was touched. `details`: `active_requests`, `queued`, `generating`. A status that cannot be read counts as busy. |
| 409 | `download_required` | Needs files from the Hub; see `allow_download`. |
| 409 | `different_model_active` | Start for another model while one runs; use switch. |
| 409 | `external_instance` | A Splash this app did not start owns the port. Not touched. |
| 409 | `port_in_use` | Another program listens on the port. Not touched. |
| 409 | `not_managed` | Stop or switch of a process the app does not own. |
| 409 | `busy` | A start, stop or switch is in progress. |
| 422 | `option_unsupported`, `exposure_unsupported` | The installed Splash lacks a flag. |
| 424 | `splash_not_installed` | No Splash executable found. |

## Typical AgentOS flow

1. `GET /status`. If `state` is `ready` and `loaded_model_id` is wanted, use `endpoint` and go.
2. Else `POST /start {"config_id": "…", "wait_ready_seconds": 120}`. Loading the cached 27B model took about 20 s here.
3. On `download_required`: stop and ask a person; never loop with `allow_download: true`.
4. Change models only with `POST /switch`. On `active_requests` wait and retry; do not stop Splash another way.
5. Run inference against `<endpoint>/v1` (Bearer inference key when `requires_api_key`).

## Limits

* One loaded model per Splash instance. There is no per-model provider endpoint.
* A request sent directly to Splash between the idle check and the stop signal
  is not covered by the check. It gets Splash's own explicit 503 `server_shutdown`; it is not
  dropped silently.
* The reasoning default is `model_default` or `off` (`--default-reasoning-effort none`,
  verified on Qwen3.8-27B with Splash 1.3.0: `reasoning_tokens` 0). Other levels are not
  offered; their meaning depends on the model template.
