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
# Historical records keep their name; the archive index and the migration note in
# the README/CLAUDE.md may mention the old name ONCE as "formerly".
EXCLUDE_DIRS = ("docs/superpowers/", "docs/archive/")
ALLOWED_MENTIONS = {
    "README.md": 1,           # "formerly llm-gateway" once, near the title
    "CLAUDE.md": 1,           # the migration note
    "tests/test_project_name.py": 99,
}


class ProjectName(unittest.TestCase):
    def test_the_old_name_is_gone_from_everything_alive(self):
        files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
        offenders = {}
        for f in files:
            if f.startswith(EXCLUDE_DIRS) or not f.endswith((".py", ".md", ".sh", ".service", ".yaml", ".txt", ".json")):
                continue
            try:
                with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
                    n = len(OLD.findall(fh.read()))
            except UnicodeDecodeError:
                continue
            if n > ALLOWED_MENTIONS.get(f, 0):
                offenders[f] = n
        self.assertEqual(offenders, {}, f"old project name still alive in: {offenders}")

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
