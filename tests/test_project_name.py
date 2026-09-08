"""The project is AI-Hub. The old name must not survive in anything ALIVE — a stale
`llm-gateway` in deploy.sh points a deploy at a path that no longer exists, in the
unit at a WorkingDirectory that is gone, in the README at a clone URL that redirects.
None of that raises; it fails at the worst moment. History (docs/superpowers,
docs/archive) keeps the name of its time on purpose.

Run: venv/bin/python -m unittest tests.test_project_name -v
"""
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = re.compile(r"llm[-_ ]gateway", re.I)
# Historical records keep their name; the migration note in the README/CLAUDE.md and
# the README's upgrade paragraph may name it, because an operator coming from the old
# install has to be told which unit to stop and which directory to move.
EXCLUDE_DIRS = ("docs/superpowers/", "docs/archive/")
# Extensionless but very much alive — the NOTICE carries the project's own name,
# and .gitignore names paths that would go stale just as silently.
ALIVE_BASENAMES = ("NOTICE", "LICENSE", ".gitignore")
ALLOWED_MENTIONS = {
    # "formerly llm-gateway" once near the title, plus the four lines of the
    # "Upgrading an install from before 2026-09-08" paragraph under
    # "Running & deploying" (stop / mv / disable / rm the old unit).
    "README.md": 5,
    "CLAUDE.md": 1,           # the migration note
    "tests/test_project_name.py": 99,
}
# A budget alone is too weak: it lets a stale path ANYWHERE in those two files pass
# as long as the total still fits. So every surviving mention must also SIT in the
# migration note or the upgrade paragraph — nothing else may carry the old name.
MIGRATION_FILES = ("README.md", "CLAUDE.md")
MIGRATION_LINE = re.compile(
    r"formerly|Renamed from|Upgrading an install|llm-gateway\.service|/opt/llm-gateway")


class ProjectName(unittest.TestCase):
    def test_the_old_name_is_gone_from_everything_alive(self):
        # -z: NUL-separated, so a path with a space stays ONE path (str.split() on
        # the default output would tear it into two names that open() cannot find).
        out = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=True)
        files = [f for f in out.split("\0") if f]
        offenders, strays = {}, {}
        for f in files:
            if f.startswith(EXCLUDE_DIRS) or not (
                f.endswith((".py", ".md", ".sh", ".service", ".yaml", ".yml", ".txt",
                            ".json"))
                or os.path.basename(f) in ALIVE_BASENAMES
            ):
                continue
            try:
                with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
                    hits = [ln for ln in fh.read().splitlines() if OLD.search(ln)]
            # Binary, or tracked-but-deleted in the working tree: neither is a stale
            # name, and neither may turn this test into an ERROR instead of a verdict.
            except (UnicodeDecodeError, FileNotFoundError):
                continue
            n = sum(len(OLD.findall(ln)) for ln in hits)
            if n > ALLOWED_MENTIONS.get(f, 0):
                offenders[f] = n
            if f in MIGRATION_FILES:
                off_note = [ln.strip() for ln in hits if not MIGRATION_LINE.search(ln)]
                if off_note:
                    strays[f] = off_note
        self.assertEqual(offenders, {}, f"old project name still alive in: {offenders}")
        self.assertEqual(strays, {}, f"old name outside the migration note: {strays}")

    def test_the_unit_and_deploy_agree_on_the_slug(self):
        with open(os.path.join(ROOT, "deploy.sh")) as fh:
            deploy = fh.read()
        self.assertIn('DEST="/opt/ai-hub"', deploy)
        self.assertIn('SERVICE="ai-hub"', deploy)
        self.assertTrue(os.path.exists(os.path.join(ROOT, "ai-hub.service")))
        with open(os.path.join(ROOT, "ai-hub.service")) as fh:
            unit = fh.read()
        self.assertIn("WorkingDirectory=/opt/ai-hub", unit)
        self.assertIn("/opt/ai-hub/venv/bin/uvicorn main:app", unit)
