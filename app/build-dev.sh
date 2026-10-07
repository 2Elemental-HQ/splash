#!/bin/bash
# Development build. Signed with a development identity or ad hoc: NOT for distribution.
# For a distributable build use release/build-release.sh.
#   ./build-dev.sh          builds build/Splash Manager.app
#   ./build-dev.sh --dmg    also makes build/SplashManager-DEV.dmg to try the disk image layout
set -euo pipefail
source "$(dirname "$0")/release/lib.sh"
cd "$(app_root)"
use_xcode
check_version_constant

swift build -c release --product SplashManager --arch arm64
BIN=$(swift build -c release --arch arm64 --show-bin-path)
APP="build/$APP_DIR_NAME"
mkdir -p build
assemble_app "$BIN/SplashManager" "$APP" "net.2elemental.splash-manager" "$APP_NAME"

# WITH_RUNTIME=1 also stages the bundled Splash runtime (unsigned layout, upstream signatures) to try it in a dev build.
if [ "${WITH_RUNTIME:-}" = 1 ]; then release/runtime/build-runtime.sh --unsigned "$APP/Contents/Resources/Splash"; fi

IDENTITY=${SIGN_IDENTITY:-$(security find-identity -v -p codesigning | sed -n 's/.*"\(Apple Development[^"]*\)".*/\1/p' | head -1)}
codesign --force --sign "${IDENTITY:--}" --options runtime "$APP"
echo "Built $APP (development signature: ${IDENTITY:-ad hoc}; not for distribution)"

if [ "${1:-}" = "--dmg" ]; then
    STAGE=$(mktemp -d)
    cp -R "$APP" "$STAGE/"
    ln -s /Applications "$STAGE/Applications"
    write_readme "$STAGE/Read Me.txt"
    rm -f build/SplashManager-DEV.dmg
    hdiutil create -volname "$APP_NAME DEV" -srcfolder "$STAGE" -ov -format UDZO -fs HFS+ build/SplashManager-DEV.dmg >/dev/null
    rm -rf "$STAGE"
    echo "Built build/SplashManager-DEV.dmg (not signed for distribution, not notarized)"
fi
