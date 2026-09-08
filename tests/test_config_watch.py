"""The config hot reload must survive an editor's rename-save — and a symlink.

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

The fourth case is the SYMLINK one, measured 2026-09-08 with the same setup: this repo's
stub-instance harness is a directory of symlinks, so config.yaml points at a file
elsewhere. A directory watch that resolves the link first watches the TARGET's directory
only — and an editor saving the instance-dir config.yaml replaces the LINK with a regular
file, an event in the lexical parent that nobody watches: 0 detections, and from then on
the gateway watches a file nobody edits. Same silence, one indirection further out.

Run: venv/bin/python -m unittest tests.test_config_watch -v
(needs an inotify-capable TMPDIR — a tmpfs/ext4 /tmp is fine, a network or overlay mount
that swallows inotify events is not.)
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

SETTLE = 0.5        # watchfiles groups changes over `step` = 50 ms; 0.5 s is ten of those —
                    # 0.3 s can let two saves land in ONE batch on a starved box, and two
                    # saves counted as one reads exactly like the bug (assertGreaterEqual
                    # tolerates a split, never a merge)


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


class TestConfigWatchSurvivesASymlinkedConfig(unittest.TestCase):
    """A config.yaml that IS a symlink (the stub-instance harness ships a directory of
    them) has two identities, and edits arrive under either one: the operator's editor
    writes the LINK path, the bytes live at the TARGET path. Watching only the resolved
    parent misses every save made through the link — including the one that replaces the
    link with a regular file, after which the watch is aimed at a file nobody edits."""

    def test_edits_through_the_link_and_through_the_target_are_all_seen(self):
        async def run(link: str, real: str) -> int:
            hits = []
            task = asyncio.ensure_future(main.watch_config(link, lambda: hits.append(1)))
            try:
                await asyncio.sleep(SETTLE)                       # let the watch arm
                _write(real, 'api_key: "a"\nbackends: []\n')      # 1) in place, via the TARGET
                await asyncio.sleep(SETTLE)
                _rename_replace(link, 'api_key: "b"\nbackends: []\n')  # 2) editor save ON the link
                await asyncio.sleep(SETTLE)                       #    (the link is a real file now)
                with open(link, "a") as f:                        # 3) in-place append after that
                    f.write("# touched\n")
                    f.flush()
                    os.fsync(f.fileno())
                await asyncio.sleep(SETTLE)
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            return len(hits)

        with tempfile.TemporaryDirectory() as d:
            realdir = os.path.join(d, "real")
            instdir = os.path.join(d, "instance")
            os.mkdir(realdir)
            os.mkdir(instdir)
            real = os.path.join(realdir, "config.yaml")
            link = os.path.join(instdir, "config.yaml")
            _write(real, 'api_key: ""\nbackends: []\n')
            os.symlink(os.path.join("..", "real", "config.yaml"), link)
            self.assertTrue(os.path.islink(link))
            seen = asyncio.run(run(link, real))

        self.assertGreaterEqual(
            seen, 3,
            f"only {seen} of 3 saves of a SYMLINKED config.yaml were detected — a watch "
            f"on the resolved parent alone never sees the editor's save on the link "
            f"path, and once that save turns the link into a regular file the gateway "
            f"is watching a file nobody edits, silently serving the old config")


if __name__ == "__main__":
    unittest.main()
