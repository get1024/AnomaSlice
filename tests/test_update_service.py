import hashlib
import io
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch
from urllib.request import Request

from update_service import (
    LATEST_URL, RELEASES_URL, UpdateCheckWorker, UpdateDownloadWorker,
    _GitHubRedirect, select_release,
)


def release_metadata(body=b"AnomaSlice test binary"):
    tag = "v1.10.0"
    assets = [
        {
            "name": name, "state": "uploaded", "size": len(body),
            "browser_download_url": f"{RELEASES_URL}/download/{tag}/{name}",
            "digest": "sha256:" + hashlib.sha256(body).hexdigest(),
        }
        for name in ("AnomaSlice.exe", "AnomaSlice-macOS-arm64.zip", "AnomaSlice-macOS-x86_64.zip")
    ]
    return {"tag_name": tag, "draft": False, "prerelease": False, "assets": assets}


class ReleaseSelectionTests(unittest.TestCase):
    def test_numeric_version_and_matching_application_architecture(self):
        for system, arch, name in (
            ("win32", "AMD64", "AnomaSlice.exe"),
            ("darwin", "arm64", "AnomaSlice-macOS-arm64.zip"),
            ("darwin", "x86_64", "AnomaSlice-macOS-x86_64.zip"),  # Intel or Rosetta
        ):
            with self.subTest(system=system, arch=arch):
                release = select_release(release_metadata(), "1.9.99", system, arch)
                self.assertEqual((release.tag, release.asset_name), ("v1.10.0", name))
                self.assertRegex(release.sha256, r"^[0-9a-f]{64}$")
        for current in ("v1.10.0", "2.0.0"):
            self.assertIsNone(select_release(release_metadata(), current, "win32", "AMD64"))

    def test_rejects_missing_untrusted_or_incompatible_update_assets(self):
        for field, value in (
            ("digest", None), ("digest", "md5:" + "0" * 32),
            ("size", True), ("size", -1), ("state", "open"), ("name", "other.exe"),
            ("browser_download_url", "https://example.org/AnomaSlice.exe"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                data = release_metadata()
                data["assets"][0][field] = value
                select_release(data, "1.0.0", "win32", "AMD64")
        for system, arch in (("linux", "x86_64"), ("win32", "ARM64"), ("darwin", "i386")):
            with self.subTest(system=system, arch=arch), self.assertRaises(ValueError):
                select_release(release_metadata(), "1.0.0", system, arch)
        for field, value in (("draft", True), ("prerelease", True), ("tag_name", "latest")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                select_release({**release_metadata(), field: value}, "1.0.0", "win32", "AMD64")

    def test_redirects_only_to_official_https_hosts(self):
        redirect = _GitHubRedirect()
        request = Request(LATEST_URL)
        url = "https://release-assets.githubusercontent.com/blob?token=from-github"
        self.assertEqual(redirect.redirect_request(request, None, 302, "", {}, url).full_url, url)
        for url in ("http://github.com/blob", "https://github.com.other.test/blob",
                    "https://user@github.com/blob", "https://other.test/blob"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                redirect.redirect_request(request, None, 302, "", {}, url)


class UpdateWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.body = b"AnomaSlice test binary"
        self.release = select_release(release_metadata(self.body), "1.0.0", "win32", "AMD64")

    def test_check_uses_the_official_latest_endpoint_and_handles_offline_errors(self):
        worker = UpdateCheckWorker("1.0.0")
        with (patch("update_service._open", return_value=io.BytesIO(json.dumps(release_metadata()).encode())) as fetch,
              patch("update_service.sys.platform", "win32"), patch("update_service.platform.machine", return_value="AMD64")):
            worker.run()
        fetch.assert_called_once_with(LATEST_URL, "application/vnd.github+json")
        self.assertEqual(worker.release, self.release)
        self.assertEqual(worker.error, "")
        for result in (OSError("offline"), io.BytesIO(b"invalid JSON")):
            with self.subTest(result=result), patch("update_service._open", side_effect=[result]):
                worker = UpdateCheckWorker("1.0.0")
                worker.run()
                self.assertTrue(worker.error)
                self.assertIsNone(worker.release)

    def test_publishes_only_a_complete_verified_file(self):
        for body in (self.body[:-1], self.body + b"x", self.body.upper(), self.body):
            worker = UpdateDownloadWorker(self.release, self.directory)
            progress = []
            worker.progress.connect(progress.append)
            with self.subTest(body=body), patch("update_service._open", return_value=io.BytesIO(body)):
                worker.run()
                if body == self.body:
                    self.assertEqual(worker.error, "")
                    self.assertEqual(worker.path.read_bytes(), body)
                    self.assertEqual(progress[-1], 100)
                    self.assertEqual(list(self.directory.iterdir()), [worker.path])
                else:
                    self.assertIsNone(worker.path)
                    self.assertTrue(worker.error)
                    self.assertNotIn(100, progress)
                    self.assertEqual(list(self.directory.iterdir()), [])

    def test_cancelling_a_running_transfer_cleans_its_partial_file(self):
        reading, resume = Event(), Event()

        class PausedResponse(io.BytesIO):
            def read1(self, size):
                reading.set()
                resume.wait(3)
                return super().read1(size)

        worker = UpdateDownloadWorker(self.release, self.directory)
        with patch("update_service._open", return_value=PausedResponse(self.body)):
            worker.start()
            try:
                self.assertTrue(reading.wait(3))
                worker.requestInterruption()
            finally:
                resume.set()
                self.assertTrue(worker.wait(5000))
        self.assertTrue(worker.cancelled)
        self.assertEqual(worker.error, "")
        self.assertIsNone(worker.path)
        self.assertEqual(list(self.directory.iterdir()), [])
