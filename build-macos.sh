#!/bin/bash
# Build with the active environment after installing requirements-build.txt.
set -euo pipefail

if [[ "$(uname -s)" != Darwin ]]; then
    echo "Run this script on macOS." >&2
    exit 1
fi

cd "$(dirname "$0")"
project_dir="$(pwd -P)"
python_bin="${PYTHON:-python3}"
arch="$("$python_bin" -c 'import platform; print(platform.machine())')"
case "$arch" in
    arm64|x86_64) ;;
    *) echo "Unsupported Python architecture: $arch" >&2; exit 1 ;;
esac

build_dir="$project_dir/build/macos/$arch"
dist_dir="$project_dir/dist"
mkdir -p "$build_dir" "$dist_dir"
"$python_bin" -m PyInstaller \
    --clean --noconfirm --windowed --onedir \
    --name AnomaSlice --target-arch "$arch" \
    --osx-bundle-identifier io.github.get1024.anomaslice \
    --specpath "$build_dir" --workpath "$build_dir/work" --distpath "$build_dir/dist" \
    "$project_dir/main.py"

# ditto retains Qt framework symlinks and executable permissions in the zip.
app="$build_dir/dist/AnomaSlice.app"
staging="$(mktemp -d "$dist_dir/.anomaslice-macos.XXXXXXXX")"
trap 'rm -rf "$staging"' EXIT
archive="AnomaSlice-macOS-$arch.zip"
/usr/bin/ditto -c -k --sequesterRsrc --keepParent "$app" "$staging/$archive"
/usr/bin/ditto -x -k "$staging/$archive" "$staging/extracted"

# Launch just the extracted bundle, with the Cocoa platform plugin and a
# temporary home for Qt settings. Python supervises startup on macOS, where
# the standard system utilities do not include GNU timeout.
"$python_bin" - "$staging/extracted" <<'PY'
import os
from pathlib import Path
import subprocess
import sys
import tempfile

extracted = Path(sys.argv[1])
executable = extracted / "AnomaSlice.app" / "Contents" / "MacOS" / "AnomaSlice"
with tempfile.TemporaryDirectory(prefix="anomaslice-smoke-") as home:
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": home,
        "CFFIXED_USER_HOME": home,
        "TMPDIR": home,
        "QT_QPA_PLATFORM": "cocoa",
        "LANG": os.environ.get("LANG") or "en_US.UTF-8",
    }
    subprocess.run([executable, "--smoke-test"], cwd=extracted,
                   env=env, timeout=90, check=True)
PY

mv -f "$staging/$archive" "$dist_dir/$archive"
echo "Verified macOS application: $dist_dir/$archive"
