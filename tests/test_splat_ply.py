"""3DGS PLY detection for the splat viewer: a wrong answer fails silently (a splat shown as a bare
download card, or a mesh PLY fed to the splat renderer, which then shows nothing)."""
import os
import tempfile
import unittest

import admin

SPLAT = (b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
         b"property float z\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
         b"property float opacity\nproperty float scale_0\nend_header\n" + b"\0" * 64)
MESH = (b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        b"element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")


class SplatPly(unittest.TestCase):
    def _file(self, data):
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.write(fd, data)
        os.close(fd)
        self.addCleanup(os.unlink, path)
        return path

    def test_gaussian_ply_is_a_splat(self):
        self.assertTrue(admin._is_splat_ply({"name": "fox.ply"}, self._file(SPLAT)))

    def test_mesh_ply_and_other_names_are_not(self):
        self.assertFalse(admin._is_splat_ply({"name": "mesh.ply"}, self._file(MESH)))
        self.assertFalse(admin._is_splat_ply({"name": "fox.glb"}, self._file(SPLAT)))
        self.assertFalse(admin._is_splat_ply({"name": "fox.ply"}, None))
        self.assertFalse(admin._is_splat_ply({"name": "fox.ply"}, "/nonexistent/fox.ply"))

    def test_cell_is_live_skip_and_page_module_has_the_scan(self):
        cell = admin._splat_cell("/ui/playground/result/j/0", "fox.ply", " download", "width:1px")
        self.assertIn('class="splatview" data-live-skip', cell)
        self.assertIn(".splatview:not([data-init])", admin._FBX_VIEWER_JS)
        self.assertTrue(os.path.isfile(os.path.join(admin._STATIC_DIR, "gaussian-splats-3d",
                                                    "gaussian-splats-3d.module.js")))


if __name__ == "__main__":
    unittest.main()
