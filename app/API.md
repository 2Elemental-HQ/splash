# Management API, version 1

Handover contract for AgentOS (PR #106 and later). This API manages the Splash
process. It does not carry inference: AgentOS keeps calling Splash's own
OpenAI-compatible endpoint (`<endpoint>/v1`).

## Where it runs and how to reach it

The API runs inside the Splash Manager app, not inside Splash. It answers while
Splash is stopped, crashed or loading, and while the management window is
closed. It stops only when the app quits. Turn on **Settings → Open at login**
so the app is there after a reboot.

| Setting | Listens on | Use |
| --- | --- | --- |
| Management: *This Mac only* (default) | `127.0.0.1:8765` | Local tools. Not reachable from the tailnet. |
| Management: *Tailscale address and this Mac* | `127.0.0.1:8765` and `<tailscale-ip>:8765` | AgentOS on the VPS. Never `0.0.0.0`. |

The port is configurable. The transport is plain HTTP; on a tailnet the traffic
is encrypted by WireGuard. For HTTPS keep *This Mac only* and publish the loopback
port with `tailscale serve`.

On the Mac used to build this (macOS 27, Tailscale app), a connection from the Mac to its
own Tailscale address timed out. That is an observation of this setup, not a general rule.
The listener was confirmed with `lsof` to be bound to loopback and the Tailscale address
only. Reachability from another device is a separate check, for example from the VPS:

```sh
curl -sS -H "Authorization: Bearer $SPLASH_MANAGER_TOKEN" http://<mac-tailnet-name-or-ip>:8765/api/v1/status
```

## Authentication

Every route needs `Authorization: Bearer <token>`. The token is 256 random bits,
created at first run, kept in the macOS Keychain (service `net.2elemental.splash-manager`).
Copy it from **Settings → Management API**, or run
`"Splash Manager.app/Contents/MacOS/SplashManager" --print-token`. A wrong token gets 401
after a 300 ms delay. The API never returns the token or the inference API key.

## Conventions

* JSON in, JSON out, `snake_case`, ISO 8601 times.
* POST bodies accept the listed fields only. Any other field gets 400 `invalid_request`.
  There is no field for a command, an argument or a model id: `config_id` names a
  configuration that a person saved in the app.
* Errors: `{"error": {"code", "message", "details"?}}`. Bodies are limited to 16 KiB.

## Saved configuration against applied configuration

* **Saved** is what a person edited in the app: a configuration (model id, revision, options) and the
  endpoint settings (port, exposure, allowed hosts).
* **Applied** is what the running process was started with. It is recorded at start and kept across
  an app restart. `applied`, `endpoint`, monitoring and `readiness` always describe the
  applied configuration of the process, never the saved one.
* `pending_changes` lists saved changes that the running process does not have. `restart_required`
  is true then. A change of port or exposure does not move the monitoring or the reported endpoint.
* A configuration id is **not** an identity. Two runs under the same id can differ in revision or options.
  Equality is decided on the *effective spec*: model id, revision, all options, port, exposure and the
  listed allowed hosts.

## Status object

`GET /api/v1/status` (and the body of start, switch and stop):

```json
{
  "api_version": 1,
  "manager": {"version": "0.3.0", "started_at": "…"},
  "state": "stopped | starting | ready | stopping | failed",
  "ownership": "none | managed | external",
  "operation": "start | stop | switch | retry | adopt | null",
  "detail": "human text or null",
  "splash": {"installed": true, "version": "1.3.0-drain.1", "source": "bundled | installed | custom",
             "integrity": "verified | failed | null", "drain_declared": true, "problem": null},
  "config": {"id": "…", "display_name": "…", "model_id": "…", "revision": "…|null"},
  "applied": {"config_id": "…", "model_id": "…", "revision": "…|null", "options": {…},
              "port": 8000, "exposure": "loopback | all_interfaces", "allowed_hosts": [], "offline": true},
  "pending_changes": {"restart_required": false, "changes": ["max context: auto -> 64K"]},
  "drain": {"supported": true, "draining": false},
  "loaded_model_id": "model id that /v1/models lists, or null",
  "endpoint": {"bind_host": "127.0.0.1", "port": 8000, "exposure": "loopback", "requires_api_key": false,
               "local_url": "http://127.0.0.1:8000", "tailnet_url": null, "openai_base_path": "/v1", "allowed_hosts": []},
  "readiness": {"process_alive": true, "http_ready": true, "model_loaded": true},
  "activity": {"active_requests": 0, "idle": true, "switch_safe": true, "reason": null},
  "process": {"pid": 1234, "started_at": "…", "uptime_seconds": 60, "adopted": false},
  "last_error": {"code": "…", "message": "…", "at": "…"},
  "retry": {"attempt": 1, "max_attempts": 3, "next_at": "…"},
  "conflict": {"kind": "external_splash | foreign_service | unreadable_service", "pid": 1, "model_id": "…", "message": "…"}
}
```

* `splash.source`: `bundled` is the runtime inside the app (upstream Splash 1.3.0 with the drain patch, checked against its
  manifest before it is used); `installed` is a Splash on this Mac (Homebrew), which the app never changes; `custom` is a path
  from Settings. The bundled one is preferred unless Settings says otherwise. `integrity` is the bundled runtime's check;
  `failed` means it is not used and `problem` says why. `drain_declared` is what its release.json promises; `drain.supported`
  is what the running process proved.
* `config` is the running configuration (from `applied`), or the selected one when nothing runs.
* `state: "ready"`: the process runs, `/ready` answered 200 **and** `/v1/models` lists the intended model.
* `ownership: "managed"`: the app started this process, or re-adopted it after an app restart after checking
  pid, kernel start time and Splash's own `/status.instance.pid`. `"external"`: another Splash already serves
  on the port; the app only watches it and never stops it.
* `operation` is the lifecycle operation that holds the single slot (see Exclusivity).
* `drain.supported` is read from the running Splash's own `/status` (`http.draining` present). null until read.
* `activity.idle` is a snapshot. `activity.switch_safe` is **not** a snapshot: it is true only when the
  running Splash can drain, so that stop and switch cannot cut a call. Idle without drain support is
  `switch_safe: false`.

## Routes

### `GET /api/v1/status`

Optional `?wait_for=ready|stopped|…&timeout=30` (seconds, max 600) holds the response until the state is
reached, a start failed for good, or the timeout ends. Always 200 with the current status; compare `state`.

### `GET /api/v1/configs`

```json
{"configs": [{
  "id": "…", "display_name": "…", "model_id": "…", "revision": "…|null",
  "availability": "loaded | local | not_local",
  "selected": true, "active": false,
  "download_required": false, "missing": [], "incomplete": {"repo": ["tokenizer.json"]},
  "draft_source": "installed_splash | unknown", "verified_start_at": "…|null", "note": null,
  "options": {"language_only": false, "max_context": null, "max_memory": null,
              "idle_release": null, "reasoning_default": "model_default | off", "disable_ane": false}
}]}
```

* `availability` is **cache inspection**, not proof. `local`: the files that Splash needs are in the Hugging Face cache
  (config, tokenizer, every weight shard named by the index, every part of a split GGUF, and the draft). Splash
  then starts with `--offline`. `not_local`: something is absent or unreadable; `missing` and `incomplete` say what.
* `loaded`: this exact saved configuration runs now (`applied` equals the saved effective spec). A configuration
  whose saved settings differ from the running ones is `active` but not `loaded`.
* `verified_start_at`: when a start of this model and revision last reached `ready` on this Mac. null means the
  files may be complete but no start was ever proven here.
* The draft model comes from the family table of the installed Splash (`install/families.py`), matched by a
  family name found in the model id. An unmatched model has `draft_source: "unknown"` and is never reported as
  fully local: a download cannot be ruled out. Splash itself decides the family at start.

### `GET /api/v1/configs/{id}/download-estimate`

Asks the Hugging Face Hub for file sizes (the only call that leaves the Mac).
`{"config_id", "download_required", "missing": [], "estimates": [{"repository", "bytes", "error"?}]}`.

### `POST /api/v1/start`

`{"config_id": "…", "allow_download": false, "wait_ready_seconds": 0}`

* **Idempotent only for the same effective spec.** A repeat of the exact running configuration returns the
  current status; concurrent identical calls start one process.
* Same id but the saved configuration or endpoint settings changed since the process started:
  409 `configuration_changed` (`details.changes`). Use switch to restart with the saved values.
* 200 when `state` is `ready`, otherwise 202. `wait_ready_seconds` (0–600) holds the call until ready, failed or timeout.
* 409 `download_required` unless `allow_download` is true and files are missing or incomplete.

### `POST /api/v1/switch`

Same body as start. Loads another configuration, **or restarts the running one with changed saved settings**.
It drains first (see Guarantees), then starts the target. It returns at once with `state: "stopping"` (202); use
`wait_ready_seconds` or `GET /status?wait_for=ready`. Same effective spec as the running one: returns the status
unchanged (idempotent). Nothing runs: acts as start.

### `POST /api/v1/stop`

`{"wait_stopped_seconds": 0}` (optional). Drains, then stops. Returns 202 with `state: "stopping"`, or 200 once
stopped. There is no `force` over the API. A second stop while one runs joins it.

### `GET /api/v1/logs?lines=100`

The last lines (max 500) of app and Splash console output, redacted. Splash 1.3.0 prints metadata only.

## Exclusivity of lifecycle operations

Start, stop, switch, automatic restart after a crash and adoption after an app restart share one slot.
The slot is claimed before the first point where the operation can wait, and each step re-checks that it still
holds the slot and still acts on the same process (pid and kernel start time). So:

* Only one operation runs. A request that conflicts gets 409 `busy` with `details.operation`.
* Joining is allowed only for an identical request: the same start, the same switch target, a second stop.
* A late answer from an old process, probe or timer cannot change the state of a newer operation or stop a process started later.
* The only thing that replaces a running operation is a person's explicit *Stop now* in the app window.

## Guarantees for stop and switch

| Situation | Behaviour |
| --- | --- |
| Splash with drain support (this fork) | The app sends `SIGUSR1`. From that moment Splash refuses new generation requests with `503 server_draining`. Requests already accepted, including streams and waiting ones, run to their end. Splash exits when none is left. **No timeout forces a running call.** A drain can take as long as the longest call. |
| Idle Splash does not exit after a drain | Only then, after 60 s with nothing running, the app uses SIGTERM, SIGINT, SIGKILL. |
| Splash without drain support (a Homebrew install; never the bundled runtime) | `stop` and `switch` over the API return 409 `drain_unsupported`; `switch_safe` is false. The window offers a confirmed *Stop now* that can cut calls. |
| Process that never became ready | Stopped at once; no call could have been accepted. |
| Person chooses *Stop now* | `SIGTERM`, then `SIGINT` after 40 s, then `SIGKILL` after 10 s. Running calls end with `server_shutdown`. |

Evidence for the first and last row is in the PR description. The Splash change is documented in `UPSTREAM_PATCHES.md`.

## Error codes

| HTTP | `code` | Meaning |
| --- | --- | --- |
| 400 | `invalid_request` | Unknown field, bad type, bad JSON. |
| 401 | `unauthorized` | Missing or wrong token. |
| 404 | `unknown_config`, `not_found` | No such configuration or route. |
| 405 | `method_not_allowed` | |
| 409 | `busy` | Another lifecycle operation holds the slot. |
| 409 | `drain_unsupported` | The running Splash cannot drain; nothing was touched. |
| 409 | `configuration_changed` | Same id, different saved settings; use switch. |
| 409 | `download_required` | Files missing or incomplete; see `allow_download`. |
| 409 | `different_model_active` | Start for another configuration while one runs; use switch. |
| 409 | `external_instance` | A Splash this app did not start owns the port. Not touched. |
| 409 | `port_in_use` | Another program listens on the port. Not touched. |
| 409 | `not_managed` | Stop or switch of a process the app does not own. |
| 422 | `option_unsupported`, `exposure_unsupported` | The installed Splash lacks a flag. |
| 424 | `splash_not_installed` | No Splash executable found. |

## Which Splash runs

A DMG install needs no Homebrew and no build: the app carries the Splash runtime (engine, Python, server) and starts it
itself. Check `splash.source` and `splash.integrity` in the status if a deployment must be sure. A Splash that was installed
with Homebrew is left alone; if Settings prefers it, `drain.supported` is false and stop/switch return `drain_unsupported`.

## What AgentOS must do

1. `GET /configs` once and keep, per configuration, the `id` **and** the `revision` and `options` it expects.
   Do not identify a model by `loaded_model_id` alone.
2. `GET /status`. The wanted configuration runs when `applied.config_id` is its id, `applied.revision` and
   `applied.options` equal the expected ones, `pending_changes.restart_required` is false, and `state` is `ready`.
   Then use `endpoint` for inference.
3. Not running: `POST /start {"config_id", "wait_ready_seconds": 120}`. Loading the cached 27B model took about 20 s here.
   On `download_required` stop and ask a person; never retry with `allow_download: true` on its own.
4. Running something else, or the same id with `restart_required`: `POST /switch`, then poll `GET /status?wait_for=ready`.
   `busy`: wait and retry. `drain_unsupported`: report it; do not work around it.
5. A switch is a drain, so it can take as long as the longest running call. Calls that arrive during it get `503 server_draining`
   with `Retry-After: 1`; retry them after `wait_for=ready`.
6. Inference goes to `<endpoint>/v1` with the inference key (Bearer) when `requires_api_key`. The key is not in this API.

## Limits

* One loaded model per Splash instance. There is no per-model provider endpoint.
* Calls sent directly to Splash cannot be told apart from AgentOS's; a drain affects all of them.
* Thinking: only `model_default` and `off` (`--default-reasoning-effort none`, verified on Qwen3.8-27B with
  Splash 1.3.0: `reasoning_tokens` 0). Other levels are not offered.
* The check of a model's cache does not prove it starts. `verified_start_at` does, per model and revision.
