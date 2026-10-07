# Splash Manager

A native macOS menu bar app that supervises the installed `splash serve`
process, and a small authenticated API so another machine (AgentOS) can start,
watch, drain-stop and switch it. It is an unofficial companion to
[incoai/splash](https://github.com/incoai/splash); it does not replace the Splash
engine and ships no Splash code. Apache-2.0, like the rest of this repository.

Everything lives in `app/`. One small Splash server change (a *drain* for correct
stop and switch) lives outside it and is documented in [UPSTREAM_PATCHES.md](UPSTREAM_PATCHES.md).
The management API contract is [API.md](API.md).

## Requirements

| | Needs |
| --- | --- |
| This app | A Mac with **Apple silicon**, macOS 14 or later. The binary is arm64 only; **Intel Macs are not supported.** |
| Splash (the engine it manages) | Apple **M3 or newer** and macOS **26.4 or later** (README of Splash 1.3.0; Splash's own device check decides at every start). |

So the app opens on an M1 or M2 Mac, or on macOS 14–26.3, but Splash will most likely refuse to run there.
The Setup card in the window shows these checks for the Mac in front of you.

## Install

Distribution is a notarized disk image: `SplashManager-<version>-arm64.dmg`. Open it, drag the app to Applications,
open it. **The app carries its own Splash runtime**: no Homebrew, no Xcode, no manual build, no path setting, no
Gatekeeper workaround. See [release/README.md](release/README.md) for how it is made and verified.

* **What is inside:** the app, and the Splash 1.3.0 release (engine, bundled Python, server) with this fork's drain patch,
  about 240 MB unpacked. Every Mach-O file is re-signed with the same Developer ID as the app, so the whole image
  is notarized; `release.json` and `runtime-manifest.json` are written after signing and describe the files as shipped.
* **Integrity:** before the runtime is started the app recomputes the SHA-256 of the engine, the Metal library, the server,
  the launcher and every Mach-O file against that manifest (and of `release.json`). A mismatch means the runtime is not
  used. The app's own code signature covers the rest of the bundle. Provenance: the upstream archive is pinned by
  SHA-256 and its engine digest is checked before anything is re-signed.
* **Existing Homebrew installs are never touched.** The bundled runtime is preferred (it can drain); Settings can prefer the
  installed one, which cannot.
* **Models are separate.** Nothing downloads a model without a confirmation that shows its size. The first start of a model that is
  not in the Hugging Face cache asks first; models Splash already installed are imported.
* **Without the bundled runtime** (a development build) the Setup card offers `brew install incoai/tap/splash` after a confirmation.

**Release status:** see the PR for the current image, its SHA-256 and what was tested. No release is published and no
installation on a clean second Mac has been tested yet.

## Develop

Needs Xcode (Swift toolchain).

```sh
cd app
./build-dev.sh                 # build/Splash Manager.app, development-signed, not for distribution
./build-dev.sh --dmg           # also a DEV disk image, to check its layout
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift test    # no model needed; uses a fake Splash
WITH_RUNTIME=1 ./build-dev.sh  # also stages the bundled runtime (unsigned layout)
release/test-package.sh <dmg>  # end-to-end test of a built image, with the cached model
```

`SPLASH_MANAGER_HOME=<dir>` moves all app state and `SPLASH_MANAGER_KEYCHAIN_SERVICE=<name>` the Keychain
items, so a test run never touches the real settings. CI (`.github/workflows/splash-manager.yml`) builds and
tests on macOS without any secret.

## What it does

* **Menu bar:** state, start/stop, model choice, open window, quit.
* **Overview:** state, loaded model, readiness (process, HTTP, model), active requests, whether Splash can drain,
  endpoint, errors, retry status, and **saved changes that are not active yet**.
* **Models:** saved configurations (model id, optional revision, a few options), file availability, verified starts,
  download size check, import of models Splash installed.
* **Settings:** Splash path and version, ports, exposure, API keys, login item, restart policy, log persistence.
* **Logs:** last 2000 redacted lines.

## Design

| File | Role |
| --- | --- |
| `Supervisor.swift` | Process ownership, one operation slot, drain, restarts, adoption, applied configuration |
| `ManagementAPI.swift`, `HTTP.swift` | The API and its listeners (Network.framework, explicit addresses only) |
| `HFCache.swift` | Reads the Hugging Face cache: file completeness; size estimate |
| `SplashLocator.swift` | Finds Splash; reads its version, `serve` flags, family table and suggested models |
| `Requirements.swift` | Hardware and macOS checks; Homebrew-guided install (only without a bundled runtime) |
| `RuntimeBundle.swift` | Finds the bundled runtime and checks it against its manifest |
| `Spawn.swift`, `Secrets.swift`, `Logs.swift` | `posix_spawn` and pid identity, Keychain, redaction |

### Applied against saved

Starting records the **applied configuration**: model, revision, options, port, exposure, allowed hosts, bind address,
whether a key is needed. Monitoring, the reported endpoint, process control, crash restarts and adoption use it. Editing
a saved configuration or the endpoint settings while Splash runs changes nothing there; the Overview lists the pending
differences and offers *Restart with saved settings*. The record is written to disk, so after an app restart the adopted
process is described by what it really started with.

`start` is idempotent for the same *effective spec* (not the same id). The same id with different saved values is
`configuration_changed`; `switch` restarts it.

### One operation at a time

Start, stop, switch, crash restart and adoption share one slot, claimed before the first suspension point and re-checked
after each one, together with the process identity (pid + kernel start time). Overlapping requests get `busy` or join an
identical one; a late result cannot touch a newer process (tested with overlapping stop/switch and switch/switch).

### Stop and switch

A stop that ends running calls is not a stop. With drain support, stop and switch send `SIGUSR1`: Splash refuses new
generation requests (`503 server_draining`), lets every accepted call finish, then exits. No timeout forces a running
call. Without drain support (Homebrew 1.3.0) the API refuses them (`drain_unsupported`) and `switch_safe` is false; the
window offers a confirmed *Stop now*. Details and evidence: API.md and UPSTREAM_PATCHES.md.

### Process ownership

* Splash starts in its **own session** (`POSIX_SPAWN_SETSID`), output to a private 0600 file (a file, not a pipe, so an
  app crash does not make Splash fail on its next log line); the file is removed after a clean stop.
* The app signals only a process whose pid and kernel start time match what it spawned. **It never stops a process it does
  not own.** A Splash already on the port is `external`; another program gives `port_in_use`.
* A crash after ready is restarted at most 3 times (5, 20, 60 s) with the configuration that ran; a start that never became
  ready is reported, not retried.

### Models and cache

A configuration is `model id + optional revision + options`. Nothing is hardcoded; the first run imports what Splash
itself pinned. Options appear only if `splash serve --help` lists the flag, and values are validated.

* **Cache inspection is not proof of startability.** "Files in cache" means the files Splash needs are complete: config,
  tokenizer, every shard the weight index names, all parts of a split GGUF, and the draft. Only a start that reached
  `ready` is recorded as *start verified*. When the files are complete the app starts with `--offline`.
* **Draft models are followed, not copied:** the app reads `install/families.py` of the installed Splash and matches a
  family name in the model id. That match is an inference; Splash decides the family from the model's config at start.
  A model that matches no family is *unknown* and never reported as fully local. If the table cannot be read, nothing
  is claimed. Splash 1.3.0 offers no command that resolves a model's draft or lists required files without starting, so
  a version-bound source (the family table) is read at runtime instead of being copied.
* **Thinking:** only "model default" and "off" (`--default-reasoning-effort none`).

### Exposure and secrets

* Inference: `127.0.0.1` by default. "All interfaces" binds `0.0.0.0`, requires an API key (environment
  `SPLASH_API_KEY`, never argv) and adds this Mac's Tailscale IP and MagicDNS name as `--allowed-host` (Splash answers 403
  to unknown Host names). Binding the Tailscale address alone is not offered: on the Mac used here a connection to its own
  Tailscale address timed out, so readiness could not be proven there (an observation of this setup).
* Management API: loopback by default; optionally also the Tailscale address. Never `0.0.0.0`. Token in the Keychain.
* Logs: memory only by default; redacted; optional file in `~/Library/Logs/Splash Manager`.

### Not done / known limits

* The tailnet path of the management API is bound but not tested from another device.
* The disk image is not installed-tested on a clean second Mac, and nothing is published.
* GGUF completeness is judged by file names (variant substring and `-0000N-of-0000M` parts).
* A shell's `HF_TOKEN` or `HF_HUB_CACHE` is not inherited by a GUI app.
* `brew upgrade splash` while Splash runs: stop first, as Splash advises.
* No in-app updates.
