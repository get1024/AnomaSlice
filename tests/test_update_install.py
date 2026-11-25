import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile

from update_install import _MAC, _check_archive, _launch_update, launch_update


class UpdateInstallTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.archive = self.root / "release.zip"
        self.target = self.root / "User's copy.exe"
        self.target.write_bytes(b"old version")

    def archive_with(self, entries):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for path, link in entries:
                entry = zipfile.ZipInfo(path)
                entry.external_attr = (stat.S_IFLNK if link else stat.S_IFREG) << 16
                archive.writestr(entry, link or b"file")

    def test_archive_keeps_framework_links_and_rejects_escape(self):
        base = "AnomaSlice.app/Contents/"
        self.archive_with([(base + "Frameworks/name", ""), (base + "Resources/name", "../Frameworks/name")])
        _check_archive(self.archive)
        for entries in (
            [("Other.app/Contents/Info.plist", "")],
            [(base + "../outside", "")],
            [(base + "Resources/link", "../../../outside")],
            [(base + "Resources/link", "/absolute")],
            [(base + "Resources/link", "inner/link/../../../outside")],
            [(base + "Frameworks/link", "../Resources"), (base + "Frameworks/LINK/payload", "")],
            [("__MACOSX/AnomaSlice.app/._..", "")],
        ):
            with self.subTest(entries=entries):
                self.archive_with(entries)
                with self.assertRaises(ValueError):
                    _check_archive(self.archive)

    def test_hash_and_wrong_artifact_fail_before_replacing_old_version(self):
        self.archive.write_bytes(b"invalid release")
        actual = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        for digest in ("", "0" * 64, actual):
            with self.assertRaises(ValueError):
                _launch_update(self.archive, "win32", digest, self.target, os.getpid())
            self.assertEqual(self.target.read_bytes(), b"old version")
            self.assertEqual(list(self.root.glob(".anomaslice-update-*")), [])

    def test_source_runtime_refuses_to_overwrite_python(self):
        with self.assertRaises(RuntimeError):
            launch_update(self.archive, sys.platform, "a" * 64)

    @unittest.skipUnless(sys.platform == "darwin", "macOS native helper")
    def test_failed_mac_restart_restores_original_bundle(self):
        target = self.root / "原始应用.app"
        target.mkdir()
        (target / "original").touch()
        stage = self.root / ".anomaslice-update-fixture"
        bad_exe = stage / "unpacked/AnomaSlice.app/Contents/MacOS/AnomaSlice"
        bad_exe.parent.mkdir(parents=True)
        bad_exe.write_text("#!/bin/sh\nexit 1\n")
        bad_exe.chmod(0o755)
        with subprocess.Popen(["/usr/bin/true"]) as previous:
            previous.wait(timeout=5)
            env = dict(os.environ, ANOMA_UPDATE_STAGE=str(stage), ANOMA_UPDATE_TARGET=str(target),
                       ANOMA_UPDATE_PID=str(previous.pid), ANOMA_UPDATE_SMOKE="1")
            helper = subprocess.run(["/bin/bash", "-c", _MAC], env=env, timeout=15, capture_output=True)
        self.assertNotEqual(helper.returncode, 0)
        self.assertTrue((target / "original").exists())
        self.assertFalse(stage.exists())

    @unittest.skipUnless(sys.platform == "win32", "Windows native helper")
    def test_failed_restart_restores_original_exe(self):
        # where.exe is a genuine native binary and fails its '--smoke-test'.
        shutil.copyfile(Path(os.environ["SystemRoot"]) / "System32/where.exe", self.archive)
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        with subprocess.Popen([sys.executable, "-c", "pass"]) as previous:
            with _launch_update(self.archive, "win32", digest, self.target, previous.pid, smoke=True) as helper:
                previous.wait(timeout=10)
                self.assertNotEqual(helper.wait(timeout=40), 0)
        self.assertEqual(self.target.read_bytes(), b"old version")
        self.assertEqual(list(self.root.glob(".anomaslice-update-*")), [])
