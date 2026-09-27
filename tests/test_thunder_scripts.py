"""Static checks on the VM bootstrap script (runs on Thunder, not here).
run: venv/bin/python -m unittest tests.test_thunder_scripts -v"""
import os
import pathlib
import re
import subprocess
import tempfile
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



def _lib(snippet, *args, home=None):
    """Run <snippet> with the script sourced as a library (main() not run)."""
    env = dict(os.environ)
    if home:
        env["HOME"] = home
    return subprocess.run(
        ["bash", "-c", 'GW_BOOTSTRAP_LIB=1 source "$0"; shift 0; ' + snippet,
         str(P), *args],
        capture_output=True, text=True, env=env)


class FixRound1(unittest.TestCase):
    BASELINE = {"cumesh", "o_voxel", "flex_gemm", "nvdiffrast.torch",
                "nvdiffrec_render", "custom_rasterizer", "custom_rasterizer_kernel",
                "mesh_inpaint_processor"}

    def test_smoke_baseline_present_without_any_pack(self):
        # a Trellis2/Hunyuan pack that failed to install must not drop its modules
        # from the smoke test (it would end in GW:SMOKE ok + GW:DONE)
        r = _lib('NODE_DIRS=(); install_extensions >/dev/null; '
                 'printf "%s\\n" "${!SMOKE_MODS[@]}"')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(self.BASELINE <= set(r.stdout.split()), r.stdout)

    def test_short_commit_is_refused(self):
        r = subprocess.run(["bash", str(P), "1d61dcc"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("40-hex", r.stderr)

    def test_rc_autostart_backup_survives_a_rerun(self):
        with tempfile.TemporaryDirectory() as t:
            rc = pathlib.Path(t, ".bashrc")
            original = "export A=1\n~/start-comfyui.sh &\n"
            rc.write_text(original)
            for _ in range(2):
                r = _lib('disable_rc_autostart "$1"', str(rc))
                self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(pathlib.Path(t, ".bashrc.gw-bak").read_text(), original)
            self.assertEqual(rc.read_text(),
                             "export A=1\n# gw-disabled: ~/start-comfyui.sh &\n")

    def test_template_nodes_reported_not_listed_ones(self):
        with tempfile.TemporaryDirectory() as t:
            cn = pathlib.Path(t, "custom_nodes")
            for d in ("ComfyUI-Manager", "comfyui-manager", "gguf", "__pycache__"):
                (cn / d).mkdir(parents=True)
            (cn / "example_node.py.example").write_text("")
            nodes = pathlib.Path(t, "nodes.txt")
            nodes.write_text("# c\nhttps://github.com/ltdrdata/ComfyUI-Manager.git@"
                             + "a" * 40 + "\nregistry:gguf@2.9.8\n")
            r = _lib('report_template_nodes "$1" "$2"', str(cn), str(nodes))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.split("\n")[:-1], ["GW:TEMPLATE_NODE comfyui-manager"])

    def test_unsuitable_venv_removed_only_after_the_new_one_built(self):
        s = P.read_text()
        main = s[s.index("phase venv"):s.index("phase nodes")]
        self.assertLess(main.index("build_venv"), main.index('rm -rf "$d"'))
        self.assertIn('mv "$CUI/venv.gw-old" "$CUI/venv"', main)


if __name__ == "__main__":
    unittest.main()
