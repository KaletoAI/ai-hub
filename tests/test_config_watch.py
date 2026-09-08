"""The config hot reload must survive an editor's rename-save.

Why this file exists (the mechanism fails SILENTLY): `main.watch_config` is the only
thing that notices an edited config.yaml, and when it stops noticing, the gateway keeps
running — happily, on the OLD config, with no log line and no error anywhere. Measured
2026-09-08 on a fresh Debian 13 / Python 3.13 / watchfiles 1.2.0 install: watching the
config FILE by path (`awatch(CONFIG_PATH)`) detected exactly ONE of four edits. The one
it caught was the first `sed -i`, which does not write in place at all — it writes a
temp file and renames it over the original, and `stat` confirmed the inode changed.
The inotify watch stayed on the OLD inode, so nothing afterwards was seen: not an
in-place `echo >> config.yaml`, not a second `sed -i`, not a `cp good config.yaml`.
Since sed, vim and most editors save by rename, the very FIRST save silences hot reload
for the rest of the process — and an operator who then "restores" a broken edit gets a
gateway that still serves the broken one, while README and CLAUDE.md promise
"hot-reloaded on save".

So the three edits below are the three the old form got wrong: rename-replace, an
in-place append AFTER it, and a second rename-replace. Prod never hit this because its
config.yaml carries `backends: []` and everything is store-managed.

Run: venv/bin/python -m unittest tests.test_config_watch -v
"""
import asyncio
import os
import sys
import tempfile
import unittest

# `import main` reads ./config.yaml at import time — give it a minimal one in a temp cwd
# (same dance as test_err_text.py).
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (tests/ is one level down)
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

SETTLE = 0.3        # watchfiles groups changes over `step` = 50 ms; 0.3 s is six of those


def _write(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def _rename_replace(path: str, text: str) -> None:
    """Save the way sed -i, vim and most editors do: write a sibling, rename it over."""
    tmp = path + ".tmp"
    _write(tmp, text)
    os.replace(tmp, path)


class TestConfigWatchSurvivesRenameSave(unittest.TestCase):

    def test_three_saves_are_three_detections(self):
        async def run(cfg: str) -> int:
            hits = []
            task = asyncio.ensure_future(main.watch_config(cfg, lambda: hits.append(1)))
            try:
                await asyncio.sleep(SETTLE)                       # let the watch arm
                _rename_replace(cfg, 'api_key: "a"\nbackends: []\n')   # 1) editor save
                await asyncio.sleep(SETTLE)
                with open(cfg, "a") as f:                         # 2) in-place append
                    f.write("# touched\n")
                    f.flush()
                    os.fsync(f.fileno())
                await asyncio.sleep(SETTLE)
                _rename_replace(cfg, 'api_key: "b"\nbackends: []\n')   # 3) editor save again
                await asyncio.sleep(SETTLE)
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            return len(hits)

        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "config.yaml")
            _write(cfg, 'api_key: ""\nbackends: []\n')
            seen = asyncio.run(run(cfg))

        self.assertGreaterEqual(
            seen, 3,
            f"only {seen} of 3 saves were detected — a file-path watch dies with the "
            f"inode the first rename-save replaces, and the gateway then serves the "
            f"old config forever without saying so")

    def test_a_sibling_file_in_the_same_directory_is_ignored(self):
        """Watching the DIRECTORY must not turn every neighbouring file into a reload —
        the repo root also holds jobs/, store.db and the stats DBs."""
        async def run(cfg: str, other: str) -> int:
            hits = []
            task = asyncio.ensure_future(main.watch_config(cfg, lambda: hits.append(1)))
            try:
                await asyncio.sleep(SETTLE)
                _write(other, "noise\n")
                await asyncio.sleep(SETTLE)
                _rename_replace(other, "more noise\n")
                await asyncio.sleep(SETTLE)
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            return len(hits)

        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "config.yaml")
            _write(cfg, 'api_key: ""\nbackends: []\n')
            seen = asyncio.run(run(cfg, os.path.join(d, "store.db")))

        self.assertEqual(seen, 0, f"{seen} reload(s) triggered by an unrelated file")


if __name__ == "__main__":
    unittest.main()
