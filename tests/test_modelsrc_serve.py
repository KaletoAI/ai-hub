"""The read-only SSH forced command that serves the model share (ops/modelsrc-serve.sh).

It runs on the share host as `command="…"` in authorized_keys, so everything the
gateway's key can do on that box is what this script lets through. Every way it can
go wrong is SILENT from the gateway's side: a path that escapes the share hands out
/etc or another user's files as if they were model bytes, `hf-cache/token` leaks the
HF credential, a `;id` that reaches a shell is remote code execution, and a `list`
that drops the HF cache's snapshot symlinks makes a synced cache look complete while
every HF loader re-downloads (or fails) on the VM. Run against a tmp share with
subprocess, SSH_ORIGINAL_COMMAND set directly (as sshd does).

run: venv/bin/python -m unittest tests.test_modelsrc_serve -v"""
import hashlib
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "ops" / "modelsrc-serve.sh"

HUB = "hf-cache/hub/models--acme--thing"
SHA = "0123abcd" * 8
SPACED = "loras/my 'odd' lora.safetensors"


def run(cmd, root, unset=False):
    env = {"MODELSRC_ROOT": str(root), "PATH": os.environ["PATH"]}
    if not unset:
        env["SSH_ORIGINAL_COMMAND"] = cmd
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True,
                          timeout=30)


class Share(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="modelsrc-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        r = self.root = self.tmp / "share"
        self.outside = self.tmp / "outside.txt"
        self.outside.write_text("secret outside the share\n")

        def put(rel, data):
            p = r / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

        put("vae/a.bin", b"0123456789")
        put("checkpoints/nested/deep/m.safetensors", b"x" * 1234)
        put(SPACED, b"spaced")
        put("empty.bin", b"")
        put("vae/$x.bin", b"dollar")
        put(f"{HUB}/blobs/{SHA}", b"blob-bytes")
        put(f"{HUB}/refs/main", b"rev1")
        put("hf-cache/token", b"hf_SECRET")
        put("hf-cache/stored_tokens", b"hf_SECRET2")
        put("hf-cache/xet/x", b"xet")
        put("hf-cache/modules/m.py", b"mod")
        put(".hidden/x", b"hidden")
        put("vae/.dotfile", b"dot")
        put("a.log", b"log")
        put("vae/run.log", b"log")
        put("tab\there.bin", b"t")
        put("new\nline.bin", b"n")
        put("back\\slash.bin", b"b")
        snap = r / HUB / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        # the HF hub cache's own shape: relative links into ../../blobs
        os.symlink(f"../../blobs/{SHA}", snap / "model.safetensors")
        os.symlink("../outside.txt", r / "link-out")                  # escapes
        os.symlink(str(self.outside), r / "abs-link")                  # absolute
        os.symlink("vae/nothing.bin", r / "dangling")                  # dangling
        os.symlink("vae", r / "dirlink")                               # a directory
        os.symlink("hf-cache/token", r / "token-link")                 # denied target
        os.symlink("a.log", r / "log-link")                            # denied target
        os.symlink("../vae/../vae/a.bin", r / "checkpoints" / "dotty")  # .. mid-path
        os.symlink("./a.bin", r / "vae" / "dot-link")                  # dot segment
        os.symlink("../" * 6 + "outside.txt", snap / "up-too-far")      # climbs out
        os.symlink("../vae/a.bin", r / "checkpoints" / "vae-link")     # plain in-tree
        os.symlink("vae-link", r / "checkpoints" / "chain")            # link → link
        os.symlink("../dirlink/a.bin", r / "checkpoints" / "via-dirlink")


class ListVerb(Share):
    def listing(self):
        p = run("list", self.root)
        self.assertEqual(p.returncode, 0, p.stderr)
        files, links = {}, {}
        for line in p.stdout.decode().splitlines():
            kind, rel, val = line.split("\t")
            if kind == "F":
                files[rel] = int(val)
            elif kind == "L":
                links[rel] = val
            else:
                self.fail(f"unknown line kind {line!r}")
        return files, links

    def test_files_with_sizes(self):
        files, _ = self.listing()
        self.assertEqual(files, {
            "vae/a.bin": 10,
            "checkpoints/nested/deep/m.safetensors": 1234,
            SPACED: 6,
            "empty.bin": 0,
            "vae/$x.bin": 6,
            f"{HUB}/blobs/{SHA}": 10,
            f"{HUB}/refs/main": 4,
        })

    def test_only_in_tree_relative_links_to_listed_files(self):
        _, links = self.listing()
        self.assertEqual(links, {
            f"{HUB}/snapshots/rev1/model.safetensors": f"../../blobs/{SHA}",
            "checkpoints/vae-link": "../vae/a.bin",
        })

    def test_every_link_target_is_a_listed_file(self):
        # the contract Task 15 relies on: recreating an L line never dangles
        files, links = self.listing()
        for rel, tgt in links.items():
            joined = os.path.normpath(os.path.join(os.path.dirname(rel), tgt))
            self.assertIn(joined, files, rel)

    def test_secrets_caches_dotfiles_logs_never_listed(self):
        out = run("list", self.root).stdout.decode()
        for bad in ("token", "stored_tokens", "xet", "modules", ".hidden",
                    ".dotfile", ".log", "tab", "line.bin", "slash"):
            self.assertNotIn(bad, out)

    def test_root_given_as_symlink(self):
        alias = self.tmp / "share-link"
        os.symlink(self.root, alias)
        p = run("list", alias)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn(b"F\tvae/a.bin\t10\n", p.stdout)

    def test_missing_root_fails_not_empty(self):
        # an empty listing would read as "the share has nothing" — a failure must say so
        p = run("list", self.tmp / "nope")
        self.assertEqual(p.returncode, 1)
        self.assertEqual(p.stdout, b"")

    @unittest.skipIf(os.geteuid() == 0, "root reads everything")
    def test_unreadable_subtree_fails_the_list(self):
        d = self.root / "locked"
        (d / "sub").mkdir(parents=True)
        (d / "sub" / "f.bin").write_bytes(b"f")
        os.chmod(d, 0)
        self.addCleanup(os.chmod, d, 0o755)
        p = run("list", self.root)
        self.assertEqual(p.returncode, 1)
        self.assertIn(b"incomplete", p.stderr)


class CatAndSha(Share):
    def test_cat_from_offset(self):
        p = run("cat vae/a.bin 3", self.root)
        self.assertEqual((p.returncode, p.stdout), (0, b"3456789"))

    def test_cat_offset_zero_and_leading_zeros(self):
        self.assertEqual(run("cat vae/a.bin 0", self.root).stdout, b"0123456789")
        # bash reads 08 as bad octal; the offset is decimal whatever it looks like
        self.assertEqual(run("cat vae/a.bin 08", self.root).stdout, b"89")

    def test_cat_past_end_is_empty(self):
        p = run("cat vae/a.bin 99", self.root)
        self.assertEqual((p.returncode, p.stdout), (0, b""))

    def test_cat_shlex_quoted_name(self):
        # the gateway shlex.quotes every remote argument
        p = run(f"cat {shlex.quote(SPACED)} 1", self.root)
        self.assertEqual((p.returncode, p.stdout), (0, b"paced"), p.stderr)

    def test_double_quotes_only_where_a_shell_would_not_expand(self):
        # shlex.quote writes '$x'; "$x" means an expansion to a shell, so it is not
        # read as the literal name
        self.assertEqual(run("cat 'vae/$x.bin' 0", self.root).stdout, b"dollar")
        p = run('cat "vae/$x.bin" 0', self.root)
        self.assertEqual((p.returncode, p.stdout), (2, b""))
        self.assertEqual(run('cat "vae/a.bin" 0', self.root).stdout, b"0123456789")

    def test_cat_hub_blob(self):
        p = run(f"cat {HUB}/blobs/{SHA} 0", self.root)
        self.assertEqual(p.stdout, b"blob-bytes")

    def test_sha256_hex_only(self):
        p = run("sha256 vae/a.bin", self.root)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.decode(),
                         hashlib.sha256(b"0123456789").hexdigest() + "\n")

    def test_sha256_quoted_name(self):
        p = run(f"sha256 {shlex.quote(SPACED)}", self.root)
        self.assertEqual(p.stdout.decode().strip(), hashlib.sha256(b"spaced").hexdigest())


class Refusals(Share):
    CASES = [
        "cat ../etc/passwd 0",
        "cat /etc/passwd 0",
        "cat link-out 0",                 # symlink out of the share
        "cat abs-link 0",
        "cat hf-cache/token 0",
        "cat hf-cache/stored_tokens 0",
        "cat hf-cache/xet/x 0",
        "cat hf-cache/modules/m.py 0",
        "cat .hidden/x 0",
        "cat vae/.dotfile 0",
        "cat a.log 0",
        "cat vae/run.log 0",
        "cat vae/a.bin -1",
        "cat vae/a.bin 1;id",
        "cat vae/a.bin 1x",
        "cat vae/a.bin 1234567890123456789",  # would overflow bash arithmetic
        "cat vae/a.bin ''",
        "rm x",
        "list; id",
        "list x",
        "list|id",
        "",
        "   ",
        "cat vae/a.bin",
        "cat vae/a.bin 0 0",
        "sha256",
        "sha256 vae/a.bin 0",
        "sha256 ../outside.txt",
        "sha256 link-out",
        "sha256 hf-cache/token",
        f"cat {HUB}/snapshots/rev1/model.safetensors 0",   # in-tree link: gateway relinks
        "cat checkpoints/vae-link 0",
        "cat dirlink/a.bin 0",           # symlinked directory on the way
        "cat vae/./a.bin 0",
        "cat ./vae/a.bin 0",
        "cat vae//a.bin 0",
        "cat vae/ 0",
        "cat vae 0",                     # a directory
        "cat vae/missing.bin 0",
        "cat 'back\\slash.bin' 0",
        "cat vae\\/a.bin 0",
        "cat $(id) 0",
        "cat `id` 0",
        "cat \"$HOME\" 0",
        "cat 'vae/a.bin 0",              # unterminated quote
        "cat vae/a.bin\t0",
        "cat vae/a.bin 0\nid",
        "cat 'tab\there.bin' 0",
        "cat vae/a.bin 0 &",
        "cat vae/a.bin 0 > /tmp/x",
        "cat * 0",
        "CAT vae/a.bin 0",
        "cat " + "a" * 5000 + " 0",
    ]

    def test_refused_with_exit_2_and_no_output(self):
        for cmd in self.CASES:
            with self.subTest(cmd=cmd[:60]):
                p = run(cmd, self.root)
                self.assertEqual(p.returncode, 2, (p.stdout, p.stderr))
                self.assertEqual(p.stdout, b"")
                self.assertIn(b"refused", p.stderr)
                self.assertNotIn(b"secret", p.stderr)

    def test_unset_command_refused(self):
        # an interactive login attempt: sshd sets no SSH_ORIGINAL_COMMAND
        p = run("", self.root, unset=True)
        self.assertEqual((p.returncode, p.stdout), (2, b""))

    def test_injection_never_runs(self):
        marker = self.tmp / "pwned"
        for cmd in (f"list; touch {marker}", f"cat vae/a.bin 0; touch {marker}",
                    f"cat $(touch {marker}) 0", f"cat 'a'$(touch {marker}) 0"):
            run(cmd, self.root)
        self.assertFalse(marker.exists())


class Hardening(Share):
    """The three hardenings of the final review: a read that cannot be redirected after
    the check, no L line across the models/ ↔ hf-cache/ split, and no silently
    half listing when hf-cache/ is a symlink."""

    def _swap_before(self, tool, rel):
        """A `<tool>` shim on PATH that swaps `rel` for a symlink out of the share and
        then runs the real tool: the TOCTOU window between the script's check and a
        read BY NAME, made deterministic. A read from the descriptor opened (and
        re-verified) before the swap is not affected by it."""
        real = shutil.which(tool)
        bindir = self.tmp / "bin"
        bindir.mkdir(exist_ok=True)
        target = self.root / rel
        shim = bindir / tool
        shim.write_text(
            "#!/bin/bash\n"
            f"touch {shlex.quote(str(self.tmp / 'swapped'))}\n"
            f"rm -f -- {shlex.quote(str(target))}\n"
            f"ln -s {shlex.quote(str(self.outside))} {shlex.quote(str(target))}\n"
            f"exec {shlex.quote(real)} \"$@\"\n")
        shim.chmod(0o755)
        return str(bindir) + os.pathsep + os.environ["PATH"]

    def _run(self, cmd, path):
        env = {"MODELSRC_ROOT": str(self.root), "PATH": path, "SSH_ORIGINAL_COMMAND": cmd}
        return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True,
                              timeout=30)

    def test_cat_reads_the_checked_file_even_if_swapped_after_the_check(self):
        path = self._swap_before("tail", "vae/a.bin")
        p = self._run("cat vae/a.bin 2", path)
        self.assertTrue((self.tmp / "swapped").exists())   # the swap really happened
        self.assertNotIn(b"secret outside", p.stdout)
        self.assertEqual((p.returncode, p.stdout), (0, b"23456789"), p.stderr)

    def test_reads_go_through_the_reverified_descriptor(self):
        # the sha256 path has no external step between check and read to hook; pin
        # the shape instead: opened once on fd 3, re-verified through it, read from it
        src = SCRIPT.read_text()
        self.assertIn('exec 3<"$ROOT_REAL/$rel"', src)
        self.assertIn("realpath -e /proc/self/fd/3", src)
        self.assertIn("-f /dev/fd/3", src)
        self.assertIn("sha256sum <&3", src)
        self.assertRegex(src, r'exec tail -c "\+\$\(\(10#\$off \+ 1\)\)" <&3')
        self.assertNotIn('-- "$ROOT_REAL/$rel"\n', src.split("case $verb in\n    list)")[1])

    def test_no_link_across_the_models_hf_cache_split(self):
        snap = self.root / HUB / "snapshots" / "rev1"
        os.symlink("../../../../../vae/a.bin", snap / "to-models")
        os.symlink(f"../{HUB}/blobs/{SHA}", self.root / "vae" / "to-hf")
        p = run("list", self.root)
        self.assertEqual(p.returncode, 0, p.stderr)
        out = p.stdout.decode()
        self.assertNotIn("to-models", out)
        self.assertNotIn("to-hf", out)
        # the same-side links are still there
        self.assertIn(f"L\t{HUB}/snapshots/rev1/model.safetensors\t", out)
        self.assertIn("L\tcheckpoints/vae-link\t", out)

    def test_symlinked_hf_cache_fails_the_list(self):
        real = self.tmp / "real-hf"
        shutil.move(str(self.root / "hf-cache"), str(real))
        os.symlink(real, self.root / "hf-cache")
        p = run("list", self.root)
        self.assertEqual(p.returncode, 1)
        self.assertEqual(p.stdout, b"")
        self.assertIn(b"hf-cache is a symlink", p.stderr)


class Static(unittest.TestCase):
    def test_parses(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_strict_mode_and_no_following(self):
        s = SCRIPT.read_text()
        self.assertTrue(re.search(r"^set -euo pipefail", s, re.M))
        self.assertIn("SSH_ORIGINAL_COMMAND", s)
        self.assertNotRegex(s, r"find[^\n]*\s-L\b")
        self.assertNotRegex(s, r"\beval\b")
        self.assertIn("-printf", s)

    def test_executable(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))


if __name__ == "__main__":
    unittest.main()
