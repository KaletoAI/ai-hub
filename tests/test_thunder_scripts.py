"""Static checks on the VM bootstrap script (runs on Thunder, not here).
run: venv/bin/python -m unittest tests.test_thunder_scripts -v"""
import pathlib
import re
import subprocess
import unittest

P = pathlib.Path("ops/thunder-bootstrap.sh")
NODES = pathlib.Path("ops/thunder-nodes.default.txt")


class Bootstrap(unittest.TestCase):
    def test_parses(self):
        subprocess.run(["bash", "-n", str(P)], check=True)

    def test_start_script_is_loop_on_localhost_with_hf_home(self):
        s = P.read_text()
        block = s[s.index("START_COMFY_EOF"):]
        self.assertIn("while :", block)
        self.assertIn("--listen 127.0.0.1", block)
        self.assertIn("HF_HOME", block)
        self.assertNotIn("0.0.0.0", s)

    def test_reports_template_models_and_smoke(self):
        s = P.read_text()
        self.assertIn("GW:UNKNOWN_MODEL", s)
        self.assertIn("GW:SMOKE", s)
        self.assertIn("GW:DONE", s)

    def test_strict_mode(self):
        self.assertTrue(re.search(r"^set -euo pipefail", P.read_text(), re.M))

    def test_handles_registry_lines(self):
        # a `registry:` line cloned as if it were a git URL fails every time
        s = P.read_text()
        self.assertIn("api.comfy.org", s)
        self.assertIn("downloadUrl", s)
        self.assertIn("GW:NODE_FAIL", s)


class DefaultNodeList(unittest.TestCase):
    GIT = re.compile(r"^https://\S+@[0-9a-f]{40}$")
    REG = re.compile(r"^registry:[A-Za-z0-9._-]+@[A-Za-z0-9._+-]+$")

    def test_every_line_is_a_pinned_form(self):
        # an unpinned or misspelled line installs whatever is current, or nothing
        live = [ln.strip() for ln in NODES.read_text().splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
        self.assertTrue(live)
        for ln in live:
            self.assertTrue(self.GIT.match(ln) or self.REG.match(ln), ln)

    def test_manager_is_in_the_list(self):
        # restart() posts /manager/reboot — without the Manager the ⟳ does nothing
        self.assertIn("ComfyUI-Manager", NODES.read_text())


def _parse(line):
    """Run the script's own line parser (sourced as a library, main() not run)."""
    out = subprocess.run(
        ["bash", "-c",
         'GW_BOOTSTRAP_LIB=1 source "$0"; rc=0; parse_node_line "$1" || rc=$?; '
         'echo "$rc|${NODE_KIND:-}|${NODE_NAME:-}|${NODE_SRC:-}|${NODE_REV:-}"',
         str(P), line],
        check=True, capture_output=True, text=True).stdout.strip()
    rc, kind, name, src, rev = out.split("|")
    return int(rc), kind, name, src, rev


class NodeLineParser(unittest.TestCase):
    def test_git_line_with_and_without_dot_git(self):
        self.assertEqual(
            _parse("https://github.com/ltdrdata/ComfyUI-Manager.git@" + "a" * 40),
            (0, "git", "ComfyUI-Manager",
             "https://github.com/ltdrdata/ComfyUI-Manager.git", "a" * 40))
        self.assertEqual(_parse("  https://github.com/leejet/ComfyUI-GGUF@abc123  ")[:3],
                         (0, "git", "ComfyUI-GGUF"))

    def test_registry_line(self):
        self.assertEqual(_parse("registry:comfyui-easy-use@1.3.6"),
                         (0, "registry", "comfyui-easy-use",
                          "registry:comfyui-easy-use", "1.3.6"))

    def test_blank_and_comment_are_skipped(self):
        self.assertEqual(_parse("")[0], 1)
        self.assertEqual(_parse("   ")[0], 1)
        self.assertEqual(_parse("# https://github.com/x/y@" + "b" * 40)[0], 1)

    def test_malformed_is_refused(self):
        # no pin, a name that walks out of custom_nodes/, a non-https source
        for bad in ("https://github.com/x/y",
                    "https://github.com/x/..@" + "c" * 40,
                    "registry:../evil@1.0",
                    "git@github.com:x/y.git@" + "d" * 40,
                    "https://github.com/x/y@rev with space"):
            self.assertEqual(_parse(bad)[0], 2, bad)

    def test_default_list_parses(self):
        for ln in NODES.read_text().splitlines():
            rc = _parse(ln)[0]
            self.assertIn(rc, (0, 1), ln)


if __name__ == "__main__":
    unittest.main()
