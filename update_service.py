"""Fetch and verify public AnomaSlice release assets in background threads."""

import hashlib
import json
import os
import platform
import re
import ssl
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

import certifi
from PySide6.QtCore import QThread, Signal


LATEST_URL = "https://api.github.com/repos/get1024/AnomaSlice/releases/latest"
RELEASES_URL = "https://github.com/get1024/AnomaSlice/releases"
_GITHUB_HOSTS = {
    "api.github.com", "github.com", "release-assets.githubusercontent.com",
    "objects.githubusercontent.com", "github-releases.githubusercontent.com",
}
_METADATA_LIMIT = 2 * 1024 * 1024


@dataclass(frozen=True)
class Release:
    tag: str
    asset_name: str
    size: int
    sha256: str
    url: str


def _version_numbers(version):
    if not isinstance(version, str) or not re.fullmatch(r"v?[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError(f"无法识别版本号：{version}")
    return tuple(int(number) for number in version.removeprefix("v").split("."))


def _asset_name(system, machine):
    arch = machine.lower()
    if system == "win32" and arch in ("amd64", "x86_64"):
        return "AnomaSlice.exe"
    if system == "darwin" and arch in ("arm64", "x86_64"):
        return f"AnomaSlice-macOS-{arch}.zip"
    raise ValueError(f"当前系统尚无对应的更新包（{system}, {machine}）。")


def select_release(data, current_version, system, machine):
    if not isinstance(data, dict) or data.get("draft") is not False or data.get("prerelease") is not False:
        raise ValueError("GitHub 未返回正式发行版。")
    tag = data.get("tag_name")
    if _version_numbers(tag) <= _version_numbers(current_version):
        return None
    name = _asset_name(system, machine)
    assets = [a for a in data.get("assets", []) if isinstance(a, dict) and a.get("name") == name]
    if len(assets) != 1 or assets[0].get("state") != "uploaded":
        raise ValueError(f"{tag} 缺少可下载的 {name}。")
    asset = assets[0]
    digest = asset.get("digest")
    if not isinstance(digest, str) or not re.fullmatch("sha256:[0-9a-fA-F]{64}", digest):
        raise ValueError("GitHub 未提供有效的 SHA-256 校验值，无法安装此版本。")
    size = asset.get("size")
    if type(size) is not int or size <= 0:
        raise ValueError("更新包的大小无效。")
    url = f"{RELEASES_URL}/download/{tag}/{name}"
    if asset.get("browser_download_url") != url:
        raise ValueError("更新包地址与官方发布地址不符。")
    return Release(tag, name, size, digest[7:].lower(), url)


class _GitHubRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        destination = urlsplit(newurl)
        if destination.scheme != "https" or destination.netloc.lower() not in _GITHUB_HOSTS:
            raise ValueError("GitHub 下载重定向到了无效地址。")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url, accept):
    context = ssl.create_default_context()
    # Frozen Python may retain the build machine's OpenSSL CA file path.
    context.load_verify_locations(cafile=certifi.where())
    opener = build_opener(HTTPSHandler(context=context), _GitHubRedirect())
    request = Request(url, headers={"User-Agent": "AnomaSlice-Updater", "Accept": accept})
    return opener.open(request, timeout=12)


class _Cancelled(Exception):
    pass


class _UpdateWorker(QThread):
    def __init__(self, parent):
        super().__init__(parent)
        self.error = ""
        self.cancelled = False

    def _check_cancelled(self):
        if self.isInterruptionRequested():
            raise _Cancelled()

    def run(self):
        try:
            self._check_cancelled()
            self._execute()
        except _Cancelled:
            self.cancelled = True
        except Exception as error:
            if self.isInterruptionRequested():
                self.cancelled = True
            elif isinstance(error, HTTPError):
                error.close()
                self.error = f"GitHub 请求失败（HTTP {error.code}）。"
            else:
                self.error = f"更新失败：{error}"


class UpdateCheckWorker(_UpdateWorker):
    def __init__(self, current_version, parent=None):
        super().__init__(parent)
        self.current_version = current_version
        self.release = None

    def _execute(self):
        body = bytearray()
        with _open(LATEST_URL, "application/vnd.github+json") as response:
            while True:
                self._check_cancelled()
                chunk = response.read1(64 * 1024)
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > _METADATA_LIMIT:
                    raise ValueError("GitHub 返回的发行信息过大。")
        self._check_cancelled()
        self.release = select_release(json.loads(body), self.current_version, sys.platform, platform.machine())


class UpdateDownloadWorker(_UpdateWorker):
    progress = Signal(int)

    def __init__(self, release, directory, parent=None):
        super().__init__(parent)
        self.release = release
        self.directory = Path(directory)
        self.path = None

    def _execute(self):
        release = self.release
        digest = hashlib.sha256()
        received = 0
        percent = 0
        partial = None
        try:
            with _open(release.url, "application/octet-stream") as response:
                self._check_cancelled()
                with NamedTemporaryFile(dir=self.directory, suffix=".part", delete=False) as output:
                    partial = Path(output.name)
                    self.progress.emit(0)
                    while True:
                        self._check_cancelled()
                        chunk = response.read1(256 * 1024)
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > release.size:
                            raise ValueError("下载内容超过发行版标注的大小。")
                        output.write(chunk)
                        digest.update(chunk)
                        completed = min(99, received * 100 // release.size)
                        if completed != percent:
                            percent = completed
                            self.progress.emit(percent)
            self._check_cancelled()
            if received != release.size or digest.hexdigest() != release.sha256:
                raise ValueError("更新包大小或 SHA-256 校验失败，请重新下载。")
            target = self.directory / release.asset_name
            os.replace(partial, target)
            partial = None
            self.path = target
            self.progress.emit(100)
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
