import base64
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dev.tests.engine.test_documents import pdf_bytes
from dev.tools import package


def stage_release(directory):
    """The release's runtime, staged in directory from the files it ships."""
    root = directory / "source"
    stage = directory / "stage"
    stage.mkdir()
    for folder, names in (
        ("server", package.SERVER_FILES),
        ("install", package.INSTALL_FILES),
        ("install/completions", package.COMPLETION_FILES),
    ):
        (root / folder).mkdir(parents=True)
        for name in names:
            shutil.copy2(package.ROOT / folder / name, root / folder / name)
    (root / "build").mkdir()
    for name in ("splash", "splash.metallib"):
        (root / "build" / name).write_bytes(b"unused CPU test fixture")
    for name in package.LICENSE_FILES:
        shutil.copy2(package.ROOT / name, root / name)
    with mock.patch.object(package, "ROOT", root):
        package.stage_runtime(stage, "test")
    return stage


class PackagedServerTests(unittest.TestCase):
    def test_staged_server_imports_and_renders_without_the_source_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = stage_release(Path(directory))
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import os, sys; sys.path.insert(0, os.getcwd()); "
                    "from server import server, documents; "
                    "file = {'file_data': sys.stdin.read()}; "
                    "budget = documents.DocumentBudget(); "
                    "print(documents.file_content(file, budget=budget)[0]['text'])",
                ],
                cwd=stage,
                input=base64.b64encode(pdf_bytes()).decode(),
                text=True,
                capture_output=True,
                timeout=40,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ALPHA 42", result.stdout)

    def test_staged_launcher_shares_the_origin_rule_without_dependencies(self):
        # As the entry point runs it, but with no site-packages: the launcher
        # refuses an origin by the server's rule before .venv exists.
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-E",
                    "-S",
                    "install/launcher.py",
                    "serve",
                    "--model",
                    "owner/repo",
                    "--allowed-origin",
                    "http://localhost/app",
                ],
                cwd=stage_release(Path(directory)),
                text=True,
                capture_output=True,
                timeout=40,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(
                "http://localhost/app is not an origin: expected a scheme and a host",
                result.stderr,
            )


if __name__ == "__main__":
    unittest.main()
