#!/bin/bash
# Builds, signs, notarizes and staples the distributable disk image.
#
#   release/build-release.sh --check   only check that signing and notarization are possible
#   release/build-release.sh           build release/out/SplashManager-<version>-arm64.dmg
#
# It fails, and does not fall back to a development or ad hoc signature, when:
#   - no "Developer ID Application" identity with its private key is in the keychain
#   - the notarytool keychain profile is missing or rejected by Apple
#
# Environment (all optional):
#   DEVELOPER_ID_APPLICATION  identity name or SHA-1 when more than one is installed
#   NOTARY_PROFILE            notarytool keychain profile (default: splash-manager-notary)
# No secret is read or printed by this script: the signing key and the notary credentials stay in the keychain.
set -euo pipefail
source "$(dirname "$0")/lib.sh"
cd "$(app_root)"
use_xcode

BUNDLE_ID="net.2elemental.splash-manager"
PROFILE=${NOTARY_PROFILE:-splash-manager-notary}
OUT="release/out"

fail() { echo "error: $*" >&2; exit 1; }

# --- 1. signing identity -------------------------------------------------
find_identity() {
    local list
    list=$(security find-identity -v -p codesigning | grep '"Developer ID Application:' || true)
    if [ -n "${DEVELOPER_ID_APPLICATION:-}" ]; then
        local hit
        hit=$(printf '%s\n' "$list" | grep -F -- "$DEVELOPER_ID_APPLICATION" || true)
        [ "$(printf '%s' "$hit" | grep -c .)" = 1 ] || fail "DEVELOPER_ID_APPLICATION does not match exactly one 'Developer ID Application' identity with a private key."
        list=$hit
    fi
    local count; count=$(printf '%s' "$list" | grep -c . || true)
    if [ "$count" = 0 ]; then
        cat >&2 <<'MSG'
error: no "Developer ID Application" signing identity (certificate with private key) was found in the keychain.
A release build is never signed with a development or ad hoc identity.

To create one (needs the Account Holder or Admin role in the Apple Developer Program):
  1. Xcode > Settings > Accounts > select your team > Manage Certificates... > + > Developer ID Application.
     (Or create a certificate signing request in Keychain Access and request "Developer ID Application"
      at developer.apple.com/account/resources/certificates, then double-click the downloaded .cer.)
  2. Check:  security find-identity -v -p codesigning | grep "Developer ID Application"
MSG
        exit 1
    fi
    [ "$count" = 1 ] || fail "more than one Developer ID Application identity; set DEVELOPER_ID_APPLICATION to the one to use."
    printf '%s\n' "$list" | sed -n 's/.*"\(Developer ID Application:[^"]*\)".*/\1/p'
}
IDENTITY=$(find_identity)
echo "Signing identity: $IDENTITY"

# --- 2. notarization credentials ----------------------------------------------
if ! xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1; then
    cat >&2 <<MSG
error: the notarytool keychain profile "$PROFILE" is missing or Apple rejected it.
Store credentials once, typing the secret yourself when it asks (it is saved in your keychain, not in a file):
  xcrun notarytool store-credentials "$PROFILE" --apple-id "<apple id email>" --team-id "<TEAM ID>"
It asks for an app-specific password: create one at account.apple.com > Sign-In and Security > App-Specific Passwords.
Alternative with an App Store Connect API key:
  xcrun notarytool store-credentials "$PROFILE" --key <AuthKey.p8> --key-id <KEY ID> --issuer <ISSUER ID>
MSG
    exit 1
fi
echo "Notary profile: $PROFILE (accepted by Apple)"

[ "${1:-}" = "--check" ] && { echo "Release preflight passed."; exit 0; }

check_version_constant
VERSION=$(app_version)

# --- 3. build ------------------------------------------------------------------
swift build -c release --product SplashManager --arch arm64
BIN=$(swift build -c release --arch arm64 --show-bin-path)
rm -rf "$OUT"; mkdir -p "$OUT"
APP="$OUT/$APP_DIR_NAME"
assemble_app "$BIN/SplashManager" "$APP" "$BUNDLE_ID" "$APP_NAME"

# --- 4. sign the app (hardened runtime, secure timestamp) -------------------------
# The bundle holds one executable and no frameworks or helpers; if any are added, sign them first, inside out.
find "$APP" -type f \( -perm -u+x -o -name '*.dylib' \) ! -path "$APP/Contents/MacOS/SplashManager" | grep -q . \
    && fail "the bundle contains additional code; extend the signing step to sign it before the app."
codesign --force --options runtime --timestamp --sign "$IDENTITY" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

# --- 5. notarize and staple the app, then the disk image ---------------------------
notarize() {
    local file=$1 json status id
    json=$(xcrun notarytool submit "$file" --keychain-profile "$PROFILE" --wait --output-format json)
    status=$(printf '%s' "$json" | plutil -extract status raw -o - -)
    id=$(printf '%s' "$json" | plutil -extract id raw -o - -)
    if [ "$status" != "Accepted" ]; then
        echo "Notarization of $file ended with status: $status (submission $id)" >&2
        xcrun notarytool log "$id" --keychain-profile "$PROFILE" >&2 || true
        exit 1
    fi
    echo "Notarized $file (submission $id)"
}
ZIP="$OUT/notarize-app.zip"
ditto -c -k --keepParent "$APP" "$ZIP"
notarize "$ZIP"
rm -f "$ZIP"
xcrun stapler staple "$APP"

DMG="$OUT/SplashManager-$VERSION-arm64.dmg"
STAGE=$(mktemp -d)
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
write_readme "$STAGE/Read Me.txt"
hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE" -ov -format UDZO -fs HFS+ "$DMG" >/dev/null
rm -rf "$STAGE"
codesign --force --timestamp --sign "$IDENTITY" "$DMG"
notarize "$DMG"
xcrun stapler staple "$DMG"

# --- 6. verify what will be shipped ---------------------------------------------------
release/verify-artifact.sh "$DMG"
( cd "$OUT" && shasum -a 256 "$(basename "$DMG")" > "$(basename "$DMG").sha256" )
echo "Done: $DMG"
echo "Not tested here: installation on a clean second Mac (see release/README.md)."
