"""Native CI check: replace an expendable installation and run its frozen smoke test."""

import hashlib
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from update_install import _BUNDLE_ID, _launch_update


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(artifact, directory):
    sandbox_home = directory / "home"
    sandbox_home.mkdir()
    os.environ.update(HOME=str(sandbox_home), USERPROFILE=str(sandbox_home),
                      CFFIXED_USER_HOME=str(sandbox_home))
    if sys.platform == "win32":
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        executable = directory / "用户的 AnomaSlice.exe"
        shutil.copyfile(Path(os.environ["SystemRoot"]) / "System32/ping.exe", executable)
        old_command = [str(executable), "-n", "3", "127.0.0.1"]
    elif sys.platform == "darwin":
        os.environ["QT_QPA_PLATFORM"] = "cocoa"
        contents = directory / "用户的 AnomaSlice.app/Contents"
        executable = contents / "MacOS/AnomaSlice"
        executable.parent.mkdir(parents=True)
        (contents / "Info.plist").write_bytes(plistlib.dumps({
            "CFBundleIdentifier": _BUNDLE_ID, "CFBundleExecutable": "AnomaSlice",
            "CFBundlePackageType": "APPL",
        }))
        executable.write_text("#!/bin/sh\nexec /bin/sleep 2\n")
        executable.chmod(0o755)
        old_command = [str(executable)]
    else:
        raise RuntimeError("Requires a native Windows or macOS build.")

    before = digest(executable)
    # Pass a disposable download; the UI removes this path before quitting.
    copy = directory / ("download" + artifact.suffix)
    shutil.copyfile(artifact, copy)
    with subprocess.Popen(old_command, stdout=subprocess.DEVNULL) as previous:
        helper = _launch_update(copy, sys.platform, digest(copy), executable, previous.pid, smoke=True)
        copy.unlink()
        previous.wait(timeout=20)  # reap it so the POSIX helper sees its PID disappear
    with helper:
        if helper.wait(timeout=150) != 0:
            raise RuntimeError("The native helper did not install and launch the release.")
    if digest(executable) == before or list(directory.glob(".anomaslice-update-*")):
        raise RuntimeError("Expected the installed program to be replaced and staging to be cleaned.")
    print("Verified in-place replacement and frozen application restart.")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="AnomaSlice update ") as folder:
        check(Path(sys.argv[1]).resolve(strict=True), Path(folder))
