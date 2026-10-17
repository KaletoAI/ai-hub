"""Guard append-only geometry and private-atlas isolation without ML models."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

WORKER_IMPORT_ERROR = None
try:
    import numpy as np
    from PIL import Image
    import scipy
except ImportError as exc:
    WORKER_IMPORT_ERROR = str(exc)

if WORKER_IMPORT_ERROR is None:
    SPEC = importlib.util.spec_from_file_location('face_core', Path(__file__).parents[1] / 'ops' / 'comfyui-face-texture' / 'face_core.py')
    core = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(core)


def fixture(path, change=None):
    """Two separately indexed triangles intentionally share every UV texel."""
    positions = np.array([[-.8, -.8, 1], [.8, -.8, 1], [0, .8, 1],
                          [-.8, -.8, 0], [.8, -.8, 0], [0, .8, 0]], dtype='<f4')
    uv = np.array([[.1, .1], [.9, .1], [.5, .9]] * 2, dtype='<f4')
    normals = np.array([[0, 0, 1]] * 6, dtype='<f4')
    indices = np.arange(6, dtype='<u4')
    png = io.BytesIO()
    Image.new('RGBA', (32, 32), (20, 40, 60, 255)).save(png, format='PNG')
    chunks = [indices.tobytes(), positions.tobytes(), uv.tobytes(), normals.tobytes(), png.getvalue()]
    binary = bytearray()
    views = []
    for chunk in chunks:
        binary.extend(b'\0' * (-len(binary) % 4))
        views.append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(chunk)})
        binary.extend(chunk)
    doc = {'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
           'nodes': [{'mesh': 0}], 'meshes': [{'primitives': [{'attributes': {'POSITION': 1, 'TEXCOORD_0': 2, 'NORMAL': 3}, 'indices': 0, 'material': 0}]}],
           'buffers': [{'byteLength': len(binary)}], 'bufferViews': views,
           'accessors': [{'bufferView': 0, 'componentType': 5125, 'count': 6, 'type': 'SCALAR'},
                         {'bufferView': 1, 'componentType': 5126, 'count': 6, 'type': 'VEC3'},
                         {'bufferView': 2, 'componentType': 5126, 'count': 6, 'type': 'VEC2'},
                         {'bufferView': 3, 'componentType': 5126, 'count': 6, 'type': 'VEC3'}],
           'images': [{'bufferView': 4, 'mimeType': 'image/png'}], 'textures': [{'source': 0}],
           'materials': [{'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}, 'baseColorFactor': [.5, 1, 1, 1], 'metallicFactor': .25, 'roughnessFactor': .8}, 'doubleSided': True}]}
    if change:
        change(doc)
    payload = json.dumps(doc).encode()
    payload += b' ' * (-len(payload) % 4)
    binary.extend(b'\0' * (-len(binary) % 4))
    raw = struct.pack('<4sII', b'glTF', 2, 28 + len(payload) + len(binary)) + struct.pack('<II', len(payload), 0x4E4F534A) + payload + struct.pack('<II', len(binary), 0x004E4942) + binary
    path.write_bytes(raw)
    return doc, bytes(binary), png.getvalue()


@unittest.skipIf(WORKER_IMPORT_ERROR is not None, "Standalone ComfyUI worker dependencies unavailable: " + str(WORKER_IMPORT_ERROR))
class PreservationTests(unittest.TestCase):
    def test_partition_keeps_every_attribute_triangle_and_original_image(self):
        """Rebuilding vertex rows or losing one partition must not silently pass."""
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'source.glb'
            original, binary, image = fixture(src)
            source = core.read_source(src)
            output = core.partition_glb(source, np.array([True, False]), image)
            doc, emitted = core.parse_glb(output)
            self.assertEqual(emitted[:len(binary)], binary)
            self.assertEqual(doc['accessors'][:4], original['accessors'])
            self.assertEqual(doc['bufferViews'][:5], original['bufferViews'])
            self.assertEqual(doc['images'][:1], original['images'])
            self.assertEqual(doc['materials'][:1], original['materials'])
            self.assertEqual(doc['nodes'], original['nodes'])
            primitives = doc['meshes'][0]['primitives']
            self.assertEqual(len(primitives), 2)
            for primitive in primitives:
                self.assertEqual(primitive['attributes'], {'POSITION': 1, 'TEXCOORD_0': 2, 'NORMAL': 3})
            np.testing.assert_array_equal(core.accessor_array(doc, emitted, primitives[0]['indices']), [3, 4, 5])
            np.testing.assert_array_equal(core.accessor_array(doc, emitted, primitives[1]['indices']), [0, 1, 2])
            self.assertEqual(primitives[0]['material'], 0)
            self.assertEqual(primitives[1]['material'], 1)
            self.assertEqual(doc['materials'][1]['pbrMetallicRoughness']['baseColorFactor'], [.5, 1, 1, 1])
            self.assertEqual(doc['materials'][1]['pbrMetallicRoughness']['roughnessFactor'], .8)

    def test_shared_uv_body_uses_original_texture_after_paint(self):
        """Painting a shared atlas must not recolor the protected body primitive."""
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'source.glb'
            dst = Path(directory) / 'result.glb'
            _, _, original_image = fixture(src)
            source = core.read_source(src)
            view = core.make_view(source['positions'], 64, 0)
            before = core.render_albedo(source, view)
            photo = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
            mask = np.ones((64, 64), dtype=np.float32)
            atlas, metrics = core.paint_atlas(source, np.array([True, False]), before, mask, photo, mask, np.array([[1, 0], [0, 1], [0, 0]]), 1)
            self.assertGreater(metrics['edited_texture_pixels'], 0)
            painted = np.asarray(Image.open(io.BytesIO(atlas)))
            np.testing.assert_array_equal(painted[16, 16, :3], [200, 150, 200])
            np.testing.assert_array_equal(painted[0, 16, :3], [200, 150, 200])
            np.testing.assert_array_equal(painted[31, 0], [20, 40, 60, 255])
            rear_view = core.make_view(source['positions'], 64, 180)
            body_before = core.render_albedo(source, rear_view)['rgb']
            body_after = core.render_albedo(source, rear_view, painted, np.array([True, False]))['rgb']
            np.testing.assert_array_equal(body_after, body_before)
            dst.write_bytes(core.partition_glb(source, np.array([True, False]), atlas))
            doc, binary = core.parse_glb(dst.read_bytes())
            body = doc['meshes'][0]['primitives'][0]
            material = doc['materials'][body['material']]
            texture = doc['textures'][material['pbrMetallicRoughness']['baseColorTexture']['index']]
            image = doc['images'][texture['source']]
            bv = doc['bufferViews'][image['bufferView']]
            self.assertEqual(binary[bv['byteOffset']:bv['byteOffset'] + bv['byteLength']], original_image)
            self.assertGreater(metrics['gutter_pixels'], 0)

    def test_projection_samples_gltf_texel_centers(self):
        """A half-texel UV shift must not misregister the portrait gradient."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.glb'
            fixture(path)
            source = core.read_source(path)
            view = {'xy': source['uv'] * 32, 'depth': source['positions'][:, 2],
                    'camera': source['positions'], 'resolution': 32}
            rendered = core.render_albedo(source, view)
            yy, xx = np.mgrid[:32, :32]
            photo = np.stack((xx, yy, np.full_like(xx, 80)), axis=2).astype(np.uint8)
            mask = np.ones((32, 32), dtype=np.float32)
            atlas, _ = core.paint_atlas(source, np.array([True, False]), rendered,
                                       mask, photo, mask, np.array([[1, 0], [0, 1], [0, 0]]), 1)
            painted = np.asarray(Image.open(io.BytesIO(atlas)))
            np.testing.assert_array_equal(painted[16, 16, :3], [32, 16, 80])

    def test_depth_gate_refuses_occluded_triangle_even_with_shared_uv(self):
        """Removing the depth test would paint the hidden back/body triangle."""
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / 'source.glb'
            fixture(src)
            source = core.read_source(src)
            rendered = core.render_albedo(source, core.make_view(source['positions'], 64, 0))
            photo = np.full((64, 64, 3), 200, dtype=np.uint8)
            mask = np.ones((64, 64), dtype=np.float32)
            with self.assertRaisesRegex(core.FaceTextureError, 'visible.*texels'):
                core.paint_atlas(source, np.array([False, True]), rendered, mask, photo, mask, np.array([[1, 0], [0, 1], [0, 0]]), 1)

    def test_unsupported_assets_are_explicitly_rejected(self):
        """Unsupported deformation, compression and external assets must not pass."""
        mutations = [lambda d: d.update(skins=[{}]), lambda d: d.update(animations=[{}]),
                     lambda d: d['images'][0].update(uri='external.png'),
                     lambda d: d['meshes'][0]['primitives'][0].update(extensions={'KHR_draco_mesh_compression': {}}),
                     lambda d: d['bufferViews'][0].update(extensions={'EXT_meshopt_compression': {}})]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.glb'
            for change in mutations:
                with self.subTest(change=change):
                    fixture(path, change)
                    with self.assertRaises(core.FaceTextureError):
                        core.read_source(path)


if __name__ == '__main__':
    unittest.main()
