"""Verify the reusable Git fixture has the committed project shape."""

import unittest
from pathlib import Path

from tests.baseline.fixture import create_small_project
from tests.pipeline.support import git, temporary_directory


class BaselineFixtureTests(unittest.TestCase):
    def test_small_project_is_committed(self):
        with temporary_directory() as directory:
            repository = create_small_project(Path(directory))
            self.assertEqual("", git(repository, "status", "--porcelain"))
            self.assertEqual(40, len(git(repository, "rev-parse", "HEAD")))
            self.assertEqual(
                ["README.md", "src/total.py", "tests/test_total.py"],
                git(repository, "ls-files").splitlines(),
            )
