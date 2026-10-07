#!/bin/bash
# Builds the Splash runtime that the app bundles: the upstream release (verified by SHA-256) with this fork's drain
# patch laid over its server, re-signed so it can be notarized with the app.
#
#   build-runtime.sh --sign "<Developer ID Application identity>" <destination>
#   build-runtime.sh --unsigned <destination>        # CI and tests: same layout and checks, upstream's signatures
#
# It does not touch any installed Splash. Signing replaces upstream's ad hoc signatures, which changes the engine bytes,
# so release.json and runtime-manifest.json are written after signing and describe the files as they ship.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
APP=$(cd "$HERE/../.." && pwd)
REPO=$(cd "$APP/.." && pwd)
MODE=${1:?usage: build-runtime.sh --sign <identity> <destination> | --unsigned <destination>}
IDENTITY=
case $MODE in
    --sign) IDENTITY=${2:?identity}; DEST=${3:?destination} ;;
    --unsigned) DEST=${2:?destination} ;;
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
fail() { echo "error: $*" >&2; exit 1; }
field() { python3 -c "import json,sys; print(json.load(open('$HERE/UPSTREAM.json'))['$1'])"; }

URL=$(field url); SHA=$(field sha256); ROOT_DIR=$(field archive_root)
CACHE="$APP/release/cache"; mkdir -p "$CACHE"
ARCHIVE="$CACHE/$(basename "$URL")"

# 1. The upstream archive, by pinned digest.
if [ -f "$ARCHIVE" ] && [ "$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)" != "$SHA" ]; then rm -f "$ARCHIVE"; fi
if [ ! -f "$ARCHIVE" ]; then
    echo "Downloading the upstream Splash runtime (about 70 MB): $URL"
    curl -fsSL -o "$ARCHIVE.part" "$URL"
    mv "$ARCHIVE.part" "$ARCHIVE"
fi
[ "$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)" = "$SHA" ] || fail "the upstream archive does not match the pinned SHA-256 ($SHA)."
echo "Upstream archive verified: $SHA"

# 2. Unpack and check what upstream's own release.json says about its engine.
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
tar -xzf "$ARCHIVE" -C "$WORK"
SRC="$WORK/$ROOT_DIR"
[ "$(shasum -a 256 "$SRC/engine/splash" | cut -d' ' -f1)" = "$(field upstream_engine_sha256)" ] || fail "the engine in the archive differs from the pinned upstream digest."
[ "$(shasum -a 256 "$SRC/engine/splash.metallib" | cut -d' ' -f1)" = "$(field upstream_metallib_sha256)" ] || fail "the metallib in the archive differs from the pinned upstream digest."

# 3. This fork may differ from the upstream release only in the overlay files.
OVERLAY=$(grep -v '^#' "$HERE/OVERLAY.txt" | grep .)
drift=0
for dir in server install; do
    while IFS= read -r rel; do
        file="$dir/${rel#"$dir"/}"
        case " $(echo $OVERLAY) " in *" $rel "*) continue ;; esac
        if [ ! -f "$REPO/$rel" ]; then echo "missing in the fork: $rel" >&2; drift=1
        elif ! cmp -s "$REPO/$rel" "$SRC/$rel"; then echo "differs from upstream 1.3.0: $rel" >&2; drift=1; fi
    done < <(cd "$SRC" && find "$dir" -type f ! -name '*.pyc' | sort)
done
[ "$drift" = 0 ] || fail "the fork's server/install files moved past the pinned upstream release. Bump release/runtime/UPSTREAM.json (and re-check the overlay), or revert."
for rel in $OVERLAY; do
    [ -f "$REPO/$rel" ] || fail "overlay file $rel is missing."
    cp "$REPO/$rel" "$SRC/$rel"
done
grep -q "SIGUSR1" "$SRC/server/server.py" || fail "the overlaid server has no drain support."

# 4. Launcher.
mkdir -p "$SRC/bin"
cp "$HERE/bin-splash" "$SRC/bin/splash"
chmod 755 "$SRC/bin/splash"

# 5. Sign every Mach-O file, with the hardened runtime and a secure timestamp.
if [ -n "$IDENTITY" ]; then
    list="$WORK/macho.txt"
    : > "$list"
    while IFS= read -r -d '' f; do
        if [ -L "$f" ]; then continue; fi
        case "$(head -c 4 "$f" | xxd -p)" in cffaedfe|cafebabe|cefaedfe) echo "$f" >> "$list" ;; esac
    done < <(find "$SRC" -type f -print0)
    echo "Signing $(wc -l < "$list" | tr -d ' ') Mach-O files"
    xargs -P 8 -I{} codesign --force --options runtime --timestamp --sign "$IDENTITY" {} < "$list" 2> "$WORK/sign.err" \
        || { cat "$WORK/sign.err" >&2; fail "signing failed."; }
    while IFS= read -r f; do
        codesign --verify --strict "$f" 2>/dev/null || fail "signature check failed for ${f#"$SRC"/}"
        info=$(codesign -dvv "$f" 2>&1)   # captured first: grep -q ending a pipe would trip pipefail
        case $info in *"Authority=Developer ID Application"*) ;; *) fail "${f#"$SRC"/} is not signed by a Developer ID Application certificate." ;; esac
        case $info in *"flags="*"runtime"*) ;; *) fail "${f#"$SRC"/} lacks the hardened runtime." ;; esac
        case $info in *"Timestamp="*) ;; *) fail "${f#"$SRC"/} has no secure timestamp." ;; esac
    done < "$list"
fi

# 6. Hashes of what ships, after signing.
python3 "$HERE/manifest.py" "$SRC" "$HERE/UPSTREAM.json"

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
mv "$SRC" "$DEST"
echo "Runtime $(field runtime_version) staged at $DEST"
