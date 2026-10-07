# Shared by build-dev.sh and release/build-release.sh. Sourced, not run.

APP_NAME="Splash Manager"
APP_DIR_NAME="Splash Manager.app"
MIN_MACOS="14.0"

app_root() { cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd; }

use_xcode() {
    if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app ]; then
        export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
    fi
}

app_version() { tr -d '[:space:]' < "$(app_root)/VERSION"; }

# The version in VERSION and the one the app reports must agree.
check_version_constant() {
    local v; v=$(app_version)
    grep -q "managerVersion = \"$v\"" "$(app_root)/Sources/SplashManagerCore/Supervisor.swift" \
        || { echo "error: VERSION ($v) differs from SplashSupervisor.managerVersion" >&2; return 1; }
}

# assemble_app <built executable> <destination .app> <bundle id> <display name>
assemble_app() {
    local exe=$1 app=$2 bundle_id=$3 name=$4 version
    version=$(app_version)
    rm -rf "$app"
    mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
    cp "$exe" "$app/Contents/MacOS/SplashManager"
    cp "$(app_root)/release/AppIcon.icns" "$app/Contents/Resources/AppIcon.icns"
    cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>$bundle_id</string>
  <key>CFBundleName</key><string>$name</string>
  <key>CFBundleDisplayName</key><string>$name</string>
  <key>CFBundleExecutable</key><string>SplashManager</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundleVersion</key><string>$version</string>
  <key>LSMinimumSystemVersion</key><string>$MIN_MACOS</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSLocalNetworkUsageDescription</key><string>Splash Manager answers management requests from your other devices on your Tailscale network, when you turn that on.</string>
  <key>NSHumanReadableCopyright</key><string>Unofficial companion app for Splash. Apache-2.0. Splash is a product of incoai.</string>
</dict></plist>
PLIST
}

# The text shown beside the app in the disk image.
write_readme() {
    cat > "$1" <<'TXT'
Splash Manager

Drag "Splash Manager" to Applications and open it. It lives in the menu bar.

Requirements
  - A Mac with Apple silicon. Intel Macs are not supported.
  - macOS 14 or later for this app.
  - To run Splash itself: Apple M3 or newer and macOS 26.4 or later
    (the requirements of Splash 1.3.0; Splash's own check decides).

What it does not include
  - Splash. The app helps you install it with Homebrew (about 235 MB).
  - Any model. A model is downloaded only after the app shows its size and you confirm.
TXT
}

# True for a Mach-O file (thin or universal).
is_macho() {
    case "$(head -c 4 "$1" | xxd -p)" in
        cffaedfe|cafebabe|cefaedfe) return 0 ;;
    esac
    return 1
}
