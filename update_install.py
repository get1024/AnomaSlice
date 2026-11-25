"""Stage a verified GitHub release, then replace the installed app after it quits."""

import errno
import hashlib
import hmac
import os
import platform as host_platform
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath


_BUNDLE_ID = "io.github.get1024.anomaslice"
_APP = "AnomaSlice.app"


def _check_bundle(bundle):
    with (bundle / "Contents" / "Info.plist").open("rb") as stream:
        info = plistlib.load(stream)
    if (info.get("CFBundleIdentifier") != _BUNDLE_ID
            or info.get("CFBundleExecutable") != "AnomaSlice"
            or info.get("CFBundlePackageType") != "APPL"):
        raise ValueError("更新包中的应用标识不匹配。")


def _check_archive(path):
    # ditto restores the framework links that PyInstaller needs. Validate paths
    # BEFORE extraction, including aliases on case-insensitive macOS volumes.
    entries, links, metadata = set(), set(), set()
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            name = item.orig_filename.rstrip("/")
            parts = name.split("/")
            if (parts[0] not in (_APP, "__MACOSX") or any(p in ("", ".", "..") for p in parts)
                    or any(c in name for c in ("\\", ":", "\0"))):
                raise ValueError("更新 ZIP 包含不安全的路径。")
            key = unicodedata.normalize("NFD", name).casefold()
            kind = stat.S_IFMT(item.external_attr >> 16)
            if key in entries or kind not in (0, stat.S_IFREG, stat.S_IFDIR, stat.S_IFLNK):
                raise ValueError("更新 ZIP 包含重复路径或无效文件类型。")
            # Check that the local extraction header agrees with this directory
            # entry before handing the archive to another ZIP implementation.
            with archive.open(item):
                pass
            entries.add(key)
            if parts[0] == "__MACOSX" and not item.is_dir():
                # ditto applies AppleDouble records to the corresponding real
                # member after removing __MACOSX/ and the '._' name prefix.
                owner = parts[1:]
                if not owner or not owner[-1].startswith("._"):
                    raise ValueError("更新 ZIP 包含无效的 macOS 元数据。")
                owner[-1] = owner[-1][2:]
                if owner[0] != _APP or owner[-1] in ("", ".", ".."):
                    raise ValueError("更新 ZIP 元数据指向无效路径。")
                metadata.add(unicodedata.normalize("NFD", "/".join(owner)).casefold())
            if kind == stat.S_IFLNK:
                if parts[0] != _APP or item.file_size > 4096:
                    raise ValueError("更新 ZIP 包含无效链接。")
                target = archive.read(item).decode("utf-8")
                steps = target.split("/")
                # Release links only ascend first, then descend into the bundle.
                # Reject a later '..' that could backtrack through another link.
                up = 0
                while up < len(steps) and steps[up] == "..":
                    up += 1
                if (up > len(parts) - 2 or any(s in ("", ".", "..") for s in steps[up:])
                        or any(c in target for c in ("\\", ":", "\0"))):
                    raise ValueError("更新 ZIP 的链接超出应用目录。")
                links.add(key)
        for key in entries:
            if any(str(p) in links for p in PurePosixPath(key).parents):
                raise ValueError("更新 ZIP 试图通过链接写入文件。")
        if not metadata.issubset(entries):
            raise ValueError("更新 ZIP 元数据缺少对应的应用文件。")


def _tool(*args):
    result = subprocess.run(args, capture_output=True, timeout=120)
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise ValueError(f"更新包验证或解包失败（{Path(args[0]).name}）：{detail}")


def _check_windows_exe(path):
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) == 64 and header[:2] == b"MZ":
            stream.seek(int.from_bytes(header[60:], "little"))
            if stream.read(6) == b"PE\0\0\x64\x86":
                return
    raise ValueError("更新文件不是 Windows x64 应用程序。")


# All file names arrive as data. Neither helper interpolates them into code.
_WINDOWS = r"""
$ErrorActionPreference = 'Stop'
$stage = $env:ANOMA_UPDATE_STAGE
$target = $env:ANOMA_UPDATE_TARGET
$new = [IO.Path]::Combine($stage, 'new.exe')
$backup = [IO.Path]::Combine($stage, 'previous.exe')
$failed = [IO.Path]::Combine($stage, 'failed.exe')
$checking = $env:ANOMA_UPDATE_SMOKE -eq '1'
function Start-App([bool]$smoke) {
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $target
    $start.WorkingDirectory = [IO.Path]::GetDirectoryName($target)
    $start.UseShellExecute = $false
    if ($smoke) { $start.Arguments = '--smoke-test' }
    $app = [Diagnostics.Process]::Start($start)
    if ($smoke) {
        if (-not $app.WaitForExit(90000)) { $app.Kill(); throw 'Startup timed out.' }
        if ($app.ExitCode -ne 0) { throw 'Startup check failed.' }
    } elseif ($app.WaitForExit(2000) -and $app.ExitCode -ne 0) { throw 'Startup failed.' }
}
try {
    $previous = Get-Process -Id ([int]$env:ANOMA_UPDATE_PID) -ErrorAction SilentlyContinue
    if ($previous -and -not $previous.WaitForExit(120000)) { throw 'AnomaSlice is still open.' }
    # A one-file bootloader or virus scanner can hold the original briefly.
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    while ($true) {
        try { [IO.File]::Replace($new, $target, $backup); break }
        catch [System.IO.IOException] {
            if ([DateTime]::UtcNow -ge $deadline -or [IO.File]::Exists($backup)) { throw }
            Start-Sleep -Milliseconds 250
        }
    }
    Start-App $checking
    try { [IO.File]::Delete($backup) } catch { }
} catch {
    $message = 'AnomaSlice 更新失败：' + $_.Exception.Message
    if ([IO.File]::Exists($backup)) {
        try {
            if ([IO.File]::Exists($target)) { [IO.File]::Replace($backup, $target, $failed) }
            else { [IO.File]::Move($backup, $target) }
            if (-not $checking) { Start-App $false }
        } catch { $message += [Environment]::NewLine + '旧版保留在 ' + $backup }
    }
    if ($checking) { [Console]::Error.WriteLine($message) }
    else { Add-Type -AssemblyName System.Windows.Forms; [void][Windows.Forms.MessageBox]::Show($message, 'AnomaSlice') }
    exit 1
} finally {
    if (-not [IO.File]::Exists($backup)) {
        [IO.File]::Delete($new)
        [IO.File]::Delete($failed)
        [IO.Directory]::Delete($stage)
    }
}
"""


_MAC = r"""
stage=$ANOMA_UPDATE_STAGE
target=$ANOMA_UPDATE_TARGET
backup="$stage/previous.app"
lock="$target.anomaslice-update-lock"
locked=0
cleanup() {
    if [ "$locked" = 1 ]; then /bin/rmdir "$lock"; fi
    if [ ! -e "$backup" ]; then /bin/rm -rf "$stage"; fi
}
trap cleanup EXIT
fail() {
    message=$1
    if [ "$ANOMA_UPDATE_SMOKE" = 1 ]; then
        echo "$message" >&2
    else
        /usr/bin/osascript -e 'on run argv' -e 'display alert "AnomaSlice" message (item 1 of argv)' -e 'end run' "$message"
    fi
    exit 1
}
for ((i=0; i<480; i++)); do
    kill -0 "$ANOMA_UPDATE_PID" 2>/dev/null || break
    /bin/sleep .25
done
if kill -0 "$ANOMA_UPDATE_PID" 2>/dev/null; then fail '应用仍在运行，请退出后重试。'; fi
/bin/mkdir "$lock" || fail '无法锁定应用。请关闭其他更新任务，并检查安装目录的写权限。'
locked=1
/bin/mv "$target" "$backup" || fail '无法替换应用。请检查安装目录的写权限。'
installed=0
if /bin/mv "$stage/unpacked/AnomaSlice.app" "$target"; then
    installed=1
    if [ "$ANOMA_UPDATE_SMOKE" = 1 ]; then
        "$target/Contents/MacOS/AnomaSlice" --smoke-test && success=1
    else
        /usr/bin/open -n "$target" && success=1
    fi
fi
if [ "${success:-0}" = 1 ]; then
    /bin/rm -rf "$backup"
    exit 0
fi
if [ "$installed" = 1 ]; then
    /bin/mv "$target" "$stage/failed.app" || fail "更新无法启动，旧版保留在 $backup"
fi
/bin/mv "$backup" "$target" || fail "恢复失败，旧版保留在 $backup"
if [ "$ANOMA_UPDATE_SMOKE" != 1 ]; then /usr/bin/open -n "$target"; fi
fail '更新无法启动，已恢复原版本。'
"""


def _launch_update(download, platform, expected_sha256, executable, pid, *, smoke=False):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256 or ""):
        raise ValueError("缺少有效的 SHA-256 更新校验值。")
    executable = Path(executable).resolve(strict=True)
    if platform == "darwin":
        target = executable.parent.parent.parent
        if (target.suffix.lower() != ".app"
                or executable != target / "Contents" / "MacOS" / "AnomaSlice"):
            raise ValueError("请从已安装的 AnomaSlice.app 运行更新。")
        if "AppTranslocation" in target.parts:
            raise ValueError("请先将 AnomaSlice.app 移到可写的应用程序目录，重新打开后再更新。")
        _check_bundle(target)
    elif platform == "win32" and executable.suffix.lower() == ".exe":
        target = executable
    else:
        raise ValueError("当前系统或安装形式不支持直接更新。")

    try:
        stage = Path(tempfile.mkdtemp(prefix=".anomaslice-update-", dir=target.parent))
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise PermissionError("安装目录不可写，请将应用移到自己有写权限的目录后再更新。") from error
        raise
    try:
        artifact = stage / ("release.zip" if platform == "darwin" else "new.exe")
        with Path(download).open("rb") as src, artifact.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        with artifact.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if not hmac.compare_digest(digest, expected_sha256.lower()):
            raise ValueError("下载文件的 SHA-256 与发布记录不符。")
        env = os.environ.copy()
        env.update(ANOMA_UPDATE_STAGE=str(stage), ANOMA_UPDATE_TARGET=str(target),
                   ANOMA_UPDATE_PID=str(pid), ANOMA_UPDATE_SMOKE="1" if smoke else "0",
                   PYINSTALLER_RESET_ENVIRONMENT="1")
        if platform == "darwin":
            _check_archive(artifact)
            unpacked = stage / "unpacked"
            _tool("/usr/bin/ditto", "-x", "-k", str(artifact), str(unpacked))
            bundle = unpacked / _APP
            for path in bundle.rglob("*"):
                if path.is_symlink() and not path.resolve(strict=True).is_relative_to(bundle):
                    raise ValueError("更新应用包含外部链接。")
            _check_bundle(bundle)
            _tool("/usr/bin/lipo", str(bundle / "Contents/MacOS/AnomaSlice"),
                  "-verify_arch", host_platform.machine())
            _tool("/usr/bin/codesign", "--verify", "--deep", "--strict", str(bundle))
            command = ["/bin/bash", "-c", _MAC]
            options = {"start_new_session": True}
        else:
            _check_windows_exe(artifact)
            powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
            command = [str(powershell), "-NoProfile", "-NonInteractive", "-Command", _WINDOWS]
            options = {"creationflags": subprocess.CREATE_NO_WINDOW}
        return subprocess.Popen(command, cwd=target.parent, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=None if smoke else subprocess.DEVNULL,
                                **options)
    except BaseException:
        shutil.rmtree(stage)
        raise


def launch_update(download: Path, platform: str, expected_sha256: str):
    """Verify and stage an install. On success the caller must quit normally."""
    if not getattr(sys, "frozen", False):
        raise RuntimeError("请在已打包的 AnomaSlice 应用中安装更新。")
    platform = {"macos": "darwin", "windows": "win32"}.get(platform, platform)
    if platform != sys.platform:
        raise ValueError("更新包与当前系统不符。")
    _launch_update(download, platform, expected_sha256, sys.executable, os.getpid())
