# Splash Manager

A native macOS menu bar app that supervises the installed `splash serve`
process, and a small authenticated API so another machine (AgentOS) can start,
watch, stop and switch it. It is an unofficial companion to
[incoai/splash](https://github.com/incoai/splash); it does not change or
replace the Splash engine, and it ships no Splash code. Apache-2.0, like the
rest of this repository.

Everything lives in `app/`. No upstream file is edited, so merging upstream
stays conflict-free.

## Build and run

Needs macOS 14+ and Xcode (for the Swift toolchain). The installed Splash comes
from Homebrew (`/opt/homebrew/bin/splash`) or any path set in Settings.

```sh
cd app
./build-app.sh                # builds and signs build/Splash Manager.app
open "build/Splash Manager.app"
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test   # 37 tests, no model needed
```

`build-app.sh` signs with the first valid codesigning identity (override with
`SIGN_IDENTITY`). A stable identity matters: the Keychain trusts the app by its
signature, so another identity makes macOS ask again for the stored secrets.

## What it does

* **Menu bar:** state, start/stop, model choice, open window, quit.
* **Overview:** state, loaded model, readiness (process, HTTP, model), active
  requests, endpoint, errors, retry status.
* **Models:** saved configurations (model id, optional revision, a few options),
  availability, download size check, import of models Splash installed.
* **Settings:** Splash path and version, ports, exposure, API keys, login item,
  restart policy, log persistence.
* **Logs:** last 2000 redacted lines.

## Design

### Structure

`SplashManagerCore` (library, no UI) holds everything that matters and is unit
tested; `SplashManager` is the SwiftUI shell.

| File | Role |
| --- | --- |
| `Supervisor.swift` | Process ownership, state machine, restarts, adoption, switch safety |
| `ManagementAPI.swift`, `HTTP.swift` | The API and its listeners (Network.framework, explicit addresses only) |
| `HFCache.swift` | Reads the Hugging Face cache: configured → local; size estimate |
| `SplashLocator.swift` | Finds Splash, reads its version and the flags `serve --help` lists |
| `Spawn.swift` | `posix_spawn`, pid identity (kernel start time), signals |
| `Secrets.swift` | Keychain; token generation; constant-time compare |
| `Logs.swift` | Redaction, in-memory ring buffer |

### Process ownership

* The app starts `splash serve` in its **own session** (`POSIX_SPAWN_SETSID`),
  stdin from `/dev/null`, output to a private file (`run/console.out`, mode 0600,
  truncated each start, removed after a clean stop). A file, not a pipe: a pipe
  would break when the app dies and Splash would fail on its next log line.
* One `@MainActor` supervisor claims its single process slot before its first
  suspension point, so concurrent starts cannot start two processes (tested with
  12 parallel starts, and with 3 parallel HTTP starts against real Splash).
* The app signals only a process whose pid and kernel start time match what it
  spawned. **It never stops a process it does not own.** A Splash already on the
  port is shown as `external`; another program on the port gives `port_in_use`.
* After an app crash or restart, a record (`run/managed.json`) lets it adopt the
  still-running Splash, but only when pid, start time and Splash's own
  `/status.instance.pid` all match. Console output of an adopted process is not followed.
* Stop: SIGTERM (Splash releases the engine), SIGINT after 40 s (stops the engine
  at once), SIGKILL 10 s later.
* Quit: stops Splash by default (asks first if calls run). Setting "Stop Splash
  when this app quits" can be turned off; the next launch adopts it.

### Readiness, crashes, switching

* `ready` = process alive **and** `/ready` is 200 **and** `/v1/models` lists the
  configured model. If that stops being true for 4 s, the state returns to `starting`.
* A start that dies before ready is reported, not retried (it would fail the
  same way). A crash after ready is restarted up to 3 times (5, 20, 60 s); the
  counter resets after 10 minutes of being ready. A manual stop cancels pending restarts.
* Active calls are read from Splash's own `/status`: `http.requests.active`, the
  scheduler (queued, prefilling, decoding, waiting) and admission queues, sampled
  twice 0.4 s apart. Unreadable status counts as busy. Stop and switch refuse on busy;
  the window offers "Stop anyway" for a person, the API does not.

### Models

A configuration is `model id + optional revision + options`. Nothing is
hardcoded; the first run imports models that Splash itself pinned in the Hugging
Face cache (`refs/splash`). Options appear only if `splash serve --help` of the
installed version lists the flag, and every value is validated before it reaches
the command line. When the files are local the app passes `--offline`, so a start
never downloads or moves to a newer commit by surprise; a download needs a
confirmation that shows the size. The matching draft repository is mapped for the
two families of Splash 1.3.0 (`install/families.py`); an unknown family is never
treated as downloaded.

**Thinking:** only "model default" and "off" (`--default-reasoning-effort none`)
are offered. The tool does not translate other values; their meaning depends on
the model template.

### Exposure and secrets

* Inference: `127.0.0.1` by default. "All interfaces" binds `0.0.0.0`, requires
  an API key (passed as `SPLASH_API_KEY` in the environment, never in argv) and
  adds this Mac's Tailscale IP and MagicDNS name as `--allowed-host`, because
  Splash answers 403 to unknown Host names. Binding only the Tailscale address
  is not offered: macOS cannot connect to its own Tailscale address, so readiness
  could not be verified.
* Management API: loopback by default; optionally also the Tailscale address.
  Never `0.0.0.0`. Token in the Keychain.
* Logs: memory only by default; redacted; optional file in
  `~/Library/Logs/Splash Manager`.

### Not done / known limits

* The tailnet path of the management API is bound and tested locally, but not
  yet tested from another tailnet device.
* GGUF "local" detection looks for a `.gguf` file name that contains the variant.
* Splash started from a login shell may use `HF_TOKEN` or `HF_HUB_CACHE`; a GUI
  app does not inherit them. Only values present in the app's own environment are passed on.
* `brew upgrade splash` while Splash runs: stop first, as Splash advises.
* No Sparkle updates, notarization or installer.
