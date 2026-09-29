"""Checks on the two VM bootstrap scripts (they run on the managed host, not here):
`ops/host-bootstrap.sh` (every host: template autostart off, the template's models and
node packs reported, flock/uv) and `ops/thunder-bootstrap.sh` (the ComfyUI part, only
with a ComfyUI service). Static checks plus their functions run in library mode
(`GW_BOOTSTRAP_LIB=1`, main() not run).
run: venv/bin/python -m unittest tests.test_thunder_scripts -v"""
import os
import pathlib
import re
import subprocess
import tempfile
import unittest

P = pathlib.Path("ops/thunder-bootstrap.sh")       # the ComfyUI part
HOST = pathlib.Path("ops/host-bootstrap.sh")        # every host, first
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

    def test_reports_smoke_and_done(self):
        # (the template's models: ops/host-bootstrap.sh, SplitScripts)
        s = P.read_text()
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


def _parse(line, script=P):
    """Run the script's own line parser (sourced as a library, main() not run)."""
    out = subprocess.run(
        ["bash", "-c",
         'GW_BOOTSTRAP_LIB=1 source "$0"; rc=0; parse_node_line "$1" || rc=$?; '
         'echo "$rc|${NODE_KIND:-}|${NODE_NAME:-}|${NODE_SRC:-}|${NODE_REV:-}"',
         str(script), line],
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



def _lib(snippet, *args, home=None, script=P, path=None):
    """Run <snippet> with the script sourced as a library (main() not run). `path` is
    put in front of PATH (stub pkill/pgrep: a test must never signal this box)."""
    env = dict(os.environ)
    if home:
        env["HOME"] = home
    if path:
        env["PATH"] = path + os.pathsep + env["PATH"]
    return subprocess.run(
        ["bash", "-c", 'GW_BOOTSTRAP_LIB=1 source "$0"; shift 0; ' + snippet,
         str(script), *args],
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
                r = _lib('disable_rc_autostart "$1"', str(rc), script=HOST)
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
            r = _lib('report_template_nodes "$1" "$2"', str(cn), str(nodes), script=HOST)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.split("\n")[:-1], ["GW:TEMPLATE_NODE comfyui-manager"])
            # no node list (a host without a ComfyUI service to bootstrap): every pack
            # the template brought is reported
            r = _lib('report_template_nodes "$1" "$2"', str(cn), str(nodes) + ".absent",
                     script=HOST)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(sorted(r.stdout.split("\n")[:-1]),
                             ["GW:TEMPLATE_NODE ComfyUI-Manager",
                              "GW:TEMPLATE_NODE comfyui-manager", "GW:TEMPLATE_NODE gguf"])

    def test_unsuitable_venv_removed_only_after_the_new_one_built(self):
        s = P.read_text()
        main = s[s.index("phase venv"):s.index("phase nodes")]
        self.assertLess(main.index("build_venv"), main.index('rm -rf "$d"'))
        self.assertIn('mv "$CUI/venv.gw-old" "$CUI/venv"', main)


def _func(script, name):
    """The text of one shell function (from `name() {` to the first `}` at column 0)."""
    s = script.read_text()
    a = s.index(f"\n{name}() {{") + 1
    return s[a:s.index("\n}\n", a) + 3]


def _stubs(t):
    """A bin dir with pkill/pgrep stubs that only log their argv to <t>/signals — the
    host bootstrap's autostart step must never signal a process on the test box."""
    b = pathlib.Path(t, "stubbin")
    b.mkdir()
    for tool in ("pkill", "pgrep"):
        f = b / tool
        f.write_text('#!/bin/sh\necho "%s $*" >>"%s"\nexit 1\n' % (tool, pathlib.Path(t, "signals")))
        f.chmod(0o755)
    return str(b)


class SplitScripts(unittest.TestCase):
    """R-W3: the host bootstrap runs on every host, the ComfyUI part only with a ComfyUI
    service — so what every host needs must not hide in the ComfyUI script, and the
    ComfyUI script must not repeat it."""

    def test_both_parse_and_are_strict(self):
        for p in (P, HOST):
            subprocess.run(["bash", "-n", str(p)], check=True)
            self.assertTrue(re.search(r"^set -euo pipefail", p.read_text(), re.M), p)

    def test_no_any_address_in_either(self):
        # Thunder's port forwarding is public without auth: nothing may listen on it
        for p in (P, HOST):
            self.assertNotIn("0.0.0.0", p.read_text(), p)

    def test_stdin_safe_pattern(self):
        # bash reads a piped script as it runs: a child reading stdin would swallow the
        # rest of the file, so main runs on the LAST line with </dev/null
        for p in (P, HOST):
            live = [ln for ln in p.read_text().splitlines()
                    if ln.strip() and not ln.lstrip().startswith("#")]
            self.assertEqual(live[-3:], ['if [ "${GW_BOOTSTRAP_LIB:-}" != 1 ]; then',
                                         '  main "$@" </dev/null', "fi"], p)

    def test_autostart_guard_only_in_the_host_script(self):
        h, c = HOST.read_text(), P.read_text()
        for needle in ("disable_rc_autostart", "start-comfyui", ".bashrc", ".profile"):
            self.assertIn(needle, h, needle)
            self.assertNotIn(needle, c, needle)

    def test_template_reports_moved_to_the_host_script(self):
        h, c = HOST.read_text(), P.read_text()
        for tag in ("GW:UNKNOWN_MODEL", "GW:TEMPLATE_NODE"):
            self.assertIn(tag, h, tag)
            self.assertNotIn(tag, c, tag)
        self.assertIn("GW:DONE", h)
        self.assertIn("GW:PHASE", h)

    def test_tools_provisioned_by_the_host_script(self):
        h, c = HOST.read_text(), P.read_text()
        self.assertIn("astral.sh/uv", h)
        self.assertIn("util-linux", h)
        self.assertNotIn("astral.sh/uv", c)          # moved, not copied

    def test_host_script_installs_no_comfy(self):
        h = HOST.read_text()
        for needle in ("git clone", "pip install", "GW:SMOKE", "start-comfy.sh",
                       "comfyanonymous"):
            self.assertNotIn(needle, h, needle)

    def test_node_parser_copies_are_identical(self):
        # a script streamed over ssh cannot source a shared file; the two copies must
        # not drift (the template report and the install would disagree on a name)
        self.assertEqual(_func(HOST, "parse_node_line"), _func(P, "parse_node_line"))
        self.assertEqual(_parse("registry:gguf@2.9.8", HOST)[:3], (0, "registry", "gguf"))


class HostBootstrapLib(unittest.TestCase):
    def test_inventory_reports_files_over_1mb(self):
        with tempfile.TemporaryDirectory() as t:
            m = pathlib.Path(t, "models", "checkpoints")
            m.mkdir(parents=True)
            with open(m / "big.safetensors", "wb") as f:
                f.truncate(3 * 1024 * 1024)
            (m / "small.txt").write_text("x")
            r = _lib('inventory_models "$1"', t, script=HOST)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.splitlines(),
                             [f"GW:UNKNOWN_MODEL models/checkpoints/big.safetensors\t{3 * 1024 * 1024}"])
            self.assertIn("1 template model file(s)", r.stderr)
            r = _lib('inventory_models "$1"', str(pathlib.Path(t, "nothing")), script=HOST)
            self.assertEqual((r.returncode, r.stdout), (0, ""), r.stderr)

    def test_find_template_comfy(self):
        with tempfile.TemporaryDirectory() as t:
            r = _lib("find_template_comfy", home=t, script=HOST)
            self.assertEqual((r.returncode, r.stdout), (0, ""), r.stderr)
            cui = pathlib.Path(t, "ComfyUI")
            cui.mkdir()
            (cui / "main.py").write_text("")
            r = _lib("find_template_comfy", home=t, script=HOST)
            self.assertEqual(r.stdout.strip(), str(cui.resolve()))

    def test_stop_template_autostart_signals_and_guards(self):
        with tempfile.TemporaryDirectory() as t:
            stub = _stubs(t)
            pathlib.Path(t, ".bashrc").write_text("~/start-comfyui.sh &\n")
            r = _lib('stop_template_autostart "$1"', "/x/ComfyUI", home=t, script=HOST,
                     path=stub)
            self.assertEqual(r.returncode, 0, r.stderr)
            sig = pathlib.Path(t, "signals").read_text()
            self.assertIn("pkill -f start-comfy", sig)
            self.assertIn("pkill -f /x/ComfyUI/main.py", sig)
            self.assertEqual(pathlib.Path(t, ".bashrc").read_text(),
                             "# gw-disabled: ~/start-comfyui.sh &\n")

    def test_ensure_tools_keeps_an_existing_uv(self):
        # re-running on the same VM downloads nothing
        with tempfile.TemporaryDirectory() as t:
            uv = pathlib.Path(t, ".local", "bin", "uv")
            uv.parent.mkdir(parents=True)
            uv.write_text("#!/bin/sh\necho uv 0.0-test\n")
            uv.chmod(0o755)
            r = _lib("ensure_tools", home=t, script=HOST, path=_stubs(t))
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("uv 0.0-test", r.stdout)
            self.assertNotIn("installing uv", r.stdout)

    def test_main_end_to_end_on_a_fake_template(self):
        # the whole host bootstrap against a fake home: the GW lines the controller
        # reads, the autostart guard, exit 0 — and no ComfyUI touched beyond reporting
        with tempfile.TemporaryDirectory() as t:
            stub = _stubs(t)
            cui = pathlib.Path(t, "ComfyUI")
            (cui / "models" / "loras").mkdir(parents=True)
            (cui / "custom_nodes" / "ComfyUI-Manager").mkdir(parents=True)
            (cui / "custom_nodes" / "extra-pack").mkdir()
            (cui / "main.py").write_text("")
            with open(cui / "models" / "loras" / "t.safetensors", "wb") as f:
                f.truncate(2 * 1024 * 1024)
            pathlib.Path(t, ".profile").write_text("start-comfyui\n")
            pathlib.Path(t, ".gw-nodes.txt").write_text(
                "https://github.com/ltdrdata/ComfyUI-Manager.git@" + "a" * 40 + "\n")
            uv = pathlib.Path(t, ".local", "bin", "uv")
            uv.parent.mkdir(parents=True)
            uv.write_text("#!/bin/sh\necho uv 0.0-test\n")
            uv.chmod(0o755)
            env = dict(os.environ, HOME=t, PATH=stub + os.pathsep + os.environ["PATH"])
            r = subprocess.run(["bash", "-s", "--"], input=HOST.read_text(),
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            gw = [ln for ln in r.stdout.splitlines() if ln.startswith("GW:")]
            self.assertEqual(gw, ["GW:PHASE locate", "GW:PHASE autostart", "GW:PHASE inventory",
                                  f"GW:UNKNOWN_MODEL models/loras/t.safetensors\t{2 * 1024 * 1024}",
                                  "GW:TEMPLATE_NODE extra-pack", "GW:PHASE tools", "GW:DONE"])
            self.assertTrue(pathlib.Path(t, ".profile").read_text().startswith("# gw-disabled"))
            self.assertEqual(sorted(p.name for p in cui.iterdir()),
                             ["custom_nodes", "main.py", "models"])

    def test_main_without_comfy_still_finishes(self):
        # the `base` template: nothing to stop or report, tools still ensured
        with tempfile.TemporaryDirectory() as t:
            stub = _stubs(t)
            uv = pathlib.Path(t, ".local", "bin", "uv")
            uv.parent.mkdir(parents=True)
            uv.write_text("#!/bin/sh\necho uv 0.0-test\n")
            uv.chmod(0o755)
            env = dict(os.environ, HOME=t, PATH=stub + os.pathsep + os.environ["PATH"])
            r = subprocess.run(["bash", "-s", "--"], input=HOST.read_text(),
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("no ComfyUI on the image", r.stdout)
            self.assertEqual(r.stdout.splitlines()[-1], "GW:DONE")
            self.assertFalse([ln for ln in r.stdout.splitlines()
                              if ln.startswith(("GW:UNKNOWN_MODEL", "GW:TEMPLATE_NODE"))])

    def test_comfy_script_needs_the_host_uv(self):
        # uv moved to the host bootstrap: without it make_venv says so instead of
        # downloading behind the host script's back
        with tempfile.TemporaryDirectory() as t:
            r = _lib('PY_VER=0.0; rc=0; make_venv "$1/v" || rc=$?; echo "rc=$rc"', t, home=t)
            self.assertIn("rc=1", r.stdout)
            self.assertIn("host-bootstrap.sh installs it", r.stderr)


if __name__ == "__main__":
    unittest.main()
