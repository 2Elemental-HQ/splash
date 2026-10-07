# Releasing Splash Manager

A release is a notarized, stapled disk image: `release/out/SplashManager-<version>-arm64.dmg`.
`build-release.sh` builds it and refuses to produce anything else: no development or ad hoc
fallback, no unsigned image. Nothing here publishes a release.

## One-time setup (on the Mac that builds releases)

You enter the secrets yourself, locally. Never paste them into a chat or a file.

1. **Developer ID Application certificate** (Account Holder or Admin role needed).
   Xcode → Settings → Accounts → your team → Manage Certificates… → **+** → *Developer ID Application*.
   Check: `security find-identity -v -p codesigning | grep "Developer ID Application"`.
   The line must show up; that means the private key is in this keychain too.
2. **Notarization credentials** in the keychain, once:
   ```sh
   xcrun notarytool store-credentials "splash-manager-notary" --apple-id "<apple id email>" --team-id "<TEAM ID>"
   ```
   It asks for an **app-specific password** (account.apple.com → Sign-In and Security →
   App-Specific Passwords). Or use an App Store Connect API key:
   `--key AuthKey_XXXX.p8 --key-id <KEY ID> --issuer <ISSUER ID>`.
3. Preflight: `release/build-release.sh --check` (talks to Apple once to test the profile).

Use another profile name with `NOTARY_PROFILE=name`, and pick between several identities with
`DEVELOPER_ID_APPLICATION="Developer ID Application: Name (TEAMID)"`.

## Build

```sh
cd app
echo 0.3.0 > VERSION         # and set SplashSupervisor.managerVersion to the same value
release/build-release.sh
```

Steps: build arm64 release → sign app (hardened runtime, secure timestamp) → notarize app
and staple → disk image with the app, an Applications link and a requirements note → sign
image → notarize and staple image → `release/verify-artifact.sh` → SHA-256 file.

## Verify a disk image

```sh
release/verify-artifact.sh SplashManager-<version>-arm64.dmg
```

Checks signature, notarization tickets, Gatekeeper (`spctl`), Developer ID authority,
hardened runtime, timestamp, team id, arm64-only, and macOS 14.0 minimum. It needs no Xcode,
only the command line tools for `xcrun stapler`.

## Clean-Mac test (not done by the script)

The script cannot prove that a second Mac accepts the app. On a Mac that never had Xcode or
this app:

1. Download the DMG with a browser (so it gets the quarantine flag), open it, drag the app to Applications.
2. Open it. Expected: at most the standard "downloaded from the Internet" confirmation, no
   "unidentified developer" or "damaged" message. Right-click → Open must not be needed.
3. Run `release/verify-artifact.sh` on the downloaded file.
4. Check the Setup card in the window shows the correct requirement results for that Mac.

## What is in the image

* **The app.** Developer ID signed, hardened runtime, arm64 only, macOS 14 or later.
* **The Splash runtime** in `Contents/Resources/Splash`: the upstream release `splash-1.3.0-arm64-macos26.tar.gz`
  (pinned by SHA-256 in `release/runtime/UPSTREAM.json`) with two files of this fork laid over it
  (`release/runtime/OVERLAY.txt`: the drain patch). `release/runtime/build-runtime.sh` downloads and verifies the archive,
  checks that every other `server/` and `install/` file of the fork still equals upstream's (it fails when the fork moved
  past the pinned release: bump `UPSTREAM.json`), signs all 67 Mach-O files (engine, Python, extension modules) with the
  hardened runtime and a secure timestamp, and writes the manifests.
* **No model.** The app downloads a model only after showing its size and asking.

### Why re-signing is safe here (the engine hash)

Upstream's `release.json` pins the engine by SHA-256 and the engine is ad hoc signed, which notarization does not accept.
Re-signing changes the engine's bytes, so a copy of upstream's hash would be wrong. The build therefore (1) checks the
engine and Metal library against upstream's published digests *before* touching them, (2) signs, (3) writes new digests of
the files as they ship, in `release.json` and `runtime-manifest.json`, and (4) the app checks those at every start and
`verify-artifact.sh` checks them again. Nothing is trusted that was not hashed after the last change.

### Bundled, not downloaded

A separate runtime package would add a download, a second signing and notarization, and a version check between two
artifacts. Bundling is one artifact, one notarization, one seal. The cost is size (about 90 MB compressed) and that a Splash
update means a new app release; the pinned `UPSTREAM.json` makes that a one-file change.

## Test a built image end to end

```sh
release/test-package.sh release/out/SplashManager-<version>-arm64.dmg
```

Needs the Splash-pinned model in the Hugging Face cache; downloads nothing. It copies the app out of the image into a
temporary folder, unmounts the image, runs the app with a minimal environment and its own state (nothing from the source
checkout), and checks: runtime found, verified and able to drain; start and readiness; a real short call; a switch during a
running stream (the stream finishes, a new call is refused with `server_draining`, a second operation is refused); the
other configuration runs; stop. It does not replace the clean-second-Mac test above.

## Test the menu

```sh
release/test-menu.sh release/out/SplashManager-<version>-arm64.dmg
```

Starts the app from a copy with `--selftest-menu`: its real `NSMenu` is presented in tracking mode and kept open for 35 s while this script
starts Splash through the API, so the supervisor's state really changes (stopped, starting, ready) under the open menu. The test fails if the menu
closed or was rebuilt by itself, if the Model submenu object was replaced, or if choosing the other model in the submenu (the action a click
triggers) was not saved. It takes focus for a moment. It is **not a physical mouse click**: synthesising one needs the Accessibility permission, and
an unattended run cannot be granted it. A physical check (hover Model while Splash starts) remains a human step.

## Continuous integration

Two workflows, both always start on a pull request (so no required check can wait for a job that never was created) and decide per job:

| Job | Runs when this changed since it last passed | Where |
| --- | --- | --- |
| Engine build and sanitizers (`engine-check`) | `runtime/`, build files, `dev/tools`, `dev/tuning`, engine tests | upstream `ci.yml` |
| Python suite 3.12/3.13/3.14 | the above, plus `server/`, `install/`, `dev/tests`, requirements | upstream `ci.yml` |
| Drain patch tests | `server/`, `install/`, server tests | `splash-manager.yml` |
| App tests and release build | `app/Sources`, `app/Tests`, `Package.swift`, `VERSION`, `build-dev.sh` | `splash-manager.yml` |
| Bundled runtime, packaging, release gate | `app/release`, the integrity code, `server/`, `install/` | `splash-manager.yml` |

"Since it last passed" is a cache marker keyed by a hash of the git blobs of the covered paths (`.github/scripts/scope.sh`), written by the
`record` job after a pull request run in which the job succeeded. It follows content, not commits, so an app-only commit on top of server changes
does not rerun the 40-minute engine and Python jobs, while any change to what they cover, or to their workflow, does. A skipped job counts as
passed. Caches can be evicted; then the job simply runs. Pushes to `main` and manual runs with *full_verification* run everything. Pushes to
other branches start nothing: the pull request does, once. Permissions stay `contents: read`; no secret is used.
