#!/bin/bash
# Checks a distribution DMG with the signing, notarization and Gatekeeper tools of macOS.
# Usable on any Mac, including one without Xcode (xcrun stapler needs the command line tools).
#   release/verify-artifact.sh SplashManager-<version>-arm64.dmg
set -uo pipefail
DMG=${1:?usage: verify-artifact.sh <dmg>}
failures=0
check() { # check <description> <command...>
    local what=$1; shift
    if out=$("$@" 2>&1); then echo "  ok    $what"; else echo "  FAIL  $what"; printf '%s\n' "$out" | sed 's/^/        /'; failures=$((failures + 1)); fi
}
echo "Disk image: $DMG"
shasum -a 256 "$DMG"
check "disk image signature is valid and strict" codesign --verify --strict --verbose=2 "$DMG"
check "disk image carries a notarization ticket" xcrun stapler validate "$DMG"
check "Gatekeeper accepts the disk image" spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG"

MOUNT=$(mktemp -d)
cleanup() { hdiutil detach "$MOUNT" -quiet >/dev/null 2>&1; rmdir "$MOUNT" 2>/dev/null; }
trap cleanup EXIT
if ! hdiutil attach "$DMG" -nobrowse -readonly -mountpoint "$MOUNT" -quiet; then echo "  FAIL  the disk image does not mount"; exit 1; fi
APP="$MOUNT/Splash Manager.app"
check "app is present" test -d "$APP"
check "app signature is valid, deep and strict" codesign --verify --deep --strict --verbose=2 "$APP"
check "app is signed by a Developer ID Application certificate" bash -c "codesign -dvv '$APP' 2>&1 | grep -q 'Authority=Developer ID Application:'"
check "app uses the hardened runtime" bash -c "codesign -dvv '$APP' 2>&1 | grep -q 'flags=.*runtime'"
check "app has a secure timestamp" bash -c "codesign -dvv '$APP' 2>&1 | grep -q '^Timestamp='"
check "app has a team identifier" bash -c "codesign -dvv '$APP' 2>&1 | grep -q '^TeamIdentifier=[A-Z0-9]\{10\}'"
check "app carries a notarization ticket" xcrun stapler validate "$APP"
check "Gatekeeper accepts the app" spctl --assess --type execute --verbose=2 "$APP"
check "app is arm64 only (no Intel code is claimed)" bash -c "[ \"\$(lipo -archs '$APP/Contents/MacOS/SplashManager')\" = arm64 ]"
check "app declares macOS 14.0 as its minimum" bash -c "[ \"\$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' '$APP/Contents/Info.plist')\" = 14.0 ]"
check "binary targets macOS 14.0" bash -c "vtool -show-build '$APP/Contents/MacOS/SplashManager' | grep -q 'minos 14.0'"
echo
if [ "$failures" = 0 ]; then echo "All checks passed."; else echo "$failures check(s) failed."; exit 1; fi
