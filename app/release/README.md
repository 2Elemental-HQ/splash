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

## What is not in the image

Splash and models. The app installs Splash through Homebrew after a confirmation, and downloads
a model only after showing its size and asking.

Why Splash is not bundled: it is about 235 MB (216 MB of that is its own Python); its engine
binary is pinned by hash in its own `release.json` and is not Developer ID signed, so re-signing
it would break that check or ship an unsigned nested binary that fails notarization; every Splash
update would need a new notarized release of this app; and the Homebrew package is the
supported channel of the Splash authors.
