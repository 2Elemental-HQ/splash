#!/bin/sh
# Builds "Splash Manager.app" into app/build. Needs Xcode (swift toolchain with the macOS SDK).
# SIGN_IDENTITY: a codesigning identity name; default is the first valid one, else ad hoc.
set -eu
cd "$(dirname "$0")"

if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app ]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi

swift build -c release --product SplashManager
BIN=$(swift build -c release --show-bin-path)

APP="build/Splash Manager.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN/SplashManager" "$APP/Contents/MacOS/SplashManager"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>net.2elemental.splash-manager</string>
  <key>CFBundleName</key><string>Splash Manager</string>
  <key>CFBundleDisplayName</key><string>Splash Manager</string>
  <key>CFBundleExecutable</key><string>SplashManager</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>Unofficial companion app. Apache-2.0. Splash is a product of incoai.</string>
</dict></plist>
PLIST

IDENTITY=${SIGN_IDENTITY:-}
if [ -z "$IDENTITY" ]; then
    IDENTITY=$(security find-identity -v -p codesigning | sed -n 's/.*"\(.*\)"/\1/p' | head -1)
fi
codesign --force --sign "${IDENTITY:--}" --options runtime "$APP"
echo "Built $APP (signed with ${IDENTITY:-ad hoc})"
