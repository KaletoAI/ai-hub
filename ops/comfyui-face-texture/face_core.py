"""CPU face-photo projection onto an unchanged static textured glTF mesh.

MediaPipe observes fresh RGB pixels. Its learned Z coordinates are never used:
visibility comes only from the source triangles and an orthographic z buffer.
The original GLB binary is an exact prefix; a private material prevents UV
sharing with protected body triangles from changing their appearance.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import distance_transform_edt, map_coordinates

FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
STABLE = (33, 133, 362, 263, 168, 1, 61, 291, 152)
_DTYPES = {5120: 'i1', 5121: 'u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}
_WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


class FaceTextureError(ValueError):
    """Unsupported input or an observation gate failed; no passthrough success."""


def parse_glb(raw):
    if len(raw) < 28:
        raise FaceTextureError('Truncated GLB')
    magic, version, length = struct.unpack_from('<4sII', raw)
    if magic != b'glTF' or version != 2 or length != len(raw):
        raise FaceTextureError('Expected a complete glTF 2.0 binary container')
    chunks = []
    offset = 12
    while offset < length:
        if offset + 8 > length:
            raise FaceTextureError('Truncated GLB chunk header')
        size, kind = struct.unpack_from('<II', raw, offset)
        offset += 8
        if size % 4 or offset + size > length:
            raise FaceTextureError('Invalid GLB chunk length')
        chunks.append((kind, raw[offset:offset + size]))
        offset += size
    if len(chunks) != 2 or [x[0] for x in chunks] != [0x4E4F534A, 0x004E4942]:
        raise FaceTextureError('Expected exactly embedded JSON and BIN GLB chunks')
    try:
        doc = json.loads(chunks[0][1])
    except (ValueError, UnicodeError) as exc:
        raise FaceTextureError('Invalid GLB JSON') from exc
    return doc, chunks[1][1]


def accessor_array(doc, binary, index):
    accessor = doc['accessors'][index]
    if accessor.get('sparse') or 'bufferView' not in accessor:
        raise FaceTextureError('Sparse or missing-buffer accessors are unsupported')
    view = doc['bufferViews'][accessor['bufferView']]
    if view.get('buffer', 0) != 0 or view.get('extensions'):
        raise FaceTextureError('External or compressed accessor buffers are unsupported')
    try:
        dtype = np.dtype(_DTYPES[accessor['componentType']])
        width = _WIDTHS[accessor['type']]
    except KeyError as exc:
        raise FaceTextureError('Unsupported accessor type') from exc
    count = int(accessor['count'])
    stride = int(view.get('byteStride', dtype.itemsize * width))
    local = int(accessor.get('byteOffset', 0))
    start = int(view.get('byteOffset', 0)) + local
    extent = (count - 1) * stride + width * dtype.itemsize if count else 0
    if count < 0 or local < 0 or stride < width * dtype.itemsize or start < 0 or local + extent > view['byteLength'] or start + extent > len(binary):
        raise FaceTextureError('Accessor falls outside its buffer view')
    values = np.ndarray((count, width), dtype=dtype, buffer=binary,
                        offset=start, strides=(stride, dtype.itemsize))
    if accessor.get('normalized') and dtype.kind in 'iu':
        values = values.astype(np.float64) / np.iinfo(dtype).max
        if dtype.kind == 'i':
            values = np.maximum(values, -1)
    return values[:, 0] if width == 1 else values


def _node_matrix(node):
    if 'matrix' in node:
        matrix = np.asarray(node['matrix'], dtype=float).reshape(4, 4, order='F')
    else:
        x, y, z, w = node.get('rotation', [0, 0, 0, 1])
        if not np.isclose(x*x + y*y + z*z + w*w, 1, atol=1e-5):
            raise FaceTextureError('Invalid node rotation quaternion')
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation @ np.diag(node.get('scale', [1, 1, 1]))
        matrix[:3, 3] = node.get('translation', [0, 0, 0])
    if not np.isfinite(matrix).all() or np.linalg.det(matrix[:3, :3]) <= 0 or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise FaceTextureError('Nonfinite, reflected or non-affine source transform')
    return matrix


def read_source(path):
    raw = Path(path).read_bytes()
    doc, binary = parse_glb(raw)
    if doc.get('asset', {}).get('version') != '2.0':
        raise FaceTextureError('Expected glTF version 2.0')
    if doc.get('skins') or doc.get('animations'):
        raise FaceTextureError('Source must be static and unskinned; animations/skins are unsupported')
    if doc.get('extensionsRequired') or doc.get('extensions'):
        raise FaceTextureError('Required/global glTF extensions are unsupported')
    buffers = doc.get('buffers', [])
    if len(buffers) != 1 or 'uri' in buffers[0] or buffers[0]['byteLength'] > len(binary):
        raise FaceTextureError('Expected one embedded geometry buffer')
    for view in doc.get('bufferViews', []):
        if view.get('extensions') or view.get('buffer', 0) != 0:
            raise FaceTextureError('Compressed/external buffer views are unsupported')
    images = []
    for image in doc.get('images', []):
        if 'uri' in image or 'bufferView' not in image or image.get('extensions') or image.get('mimeType') not in ('image/png', 'image/jpeg'):
            raise FaceTextureError('Only embedded PNG/JPEG textures are supported; external/compressed textures are unsupported')
        view = doc['bufferViews'][image['bufferView']]
        start = int(view.get('byteOffset', 0))
        stop = start + int(view['byteLength'])
        if start < 0 or stop > len(binary):
            raise FaceTextureError('Image outside embedded buffer')
        images.append(binary[start:stop])
    meshes = doc.get('meshes', [])
    if len(meshes) != 1 or len(meshes[0].get('primitives', [])) != 1:
        raise FaceTextureError('This node requires one original static textured mesh with one primitive')
    primitive = meshes[0]['primitives'][0]
    if primitive.get('mode', 4) != 4 or primitive.get('targets') or primitive.get('extensions') or meshes[0].get('weights'):
        raise FaceTextureError('Only uncompressed, undeformed TRIANGLES are supported')
    if any(name.startswith(('JOINTS_', 'WEIGHTS_', 'COLOR_')) for name in primitive['attributes']):
        raise FaceTextureError('Skinning and vertex-color assets are unsupported')
    attributes = {name: accessor_array(doc, binary, index) for name, index in primitive['attributes'].items()}
    if 'POSITION' not in attributes or 'TEXCOORD_0' not in attributes:
        raise FaceTextureError('Source requires positions and original TEXCOORD_0')
    positions, uv = attributes['POSITION'], attributes['TEXCOORD_0']
    if positions.ndim != 2 or positions.shape[1] != 3 or uv.shape != (len(positions), 2) or any(len(a) != len(positions) or not np.isfinite(a).all() for a in attributes.values()):
        raise FaceTextureError('Invalid/nonfinite or inconsistent vertex attributes')
    if len(positions) < 3 or np.any(uv < -1e-6) or np.any(uv > 1 + 1e-6):
        raise FaceTextureError('Empty mesh or tiled/out-of-range UVs are unsupported for atlas painting')
    if 'indices' in primitive:
        index_doc = doc['accessors'][primitive['indices']]
        if index_doc['componentType'] not in (5121, 5123, 5125) or index_doc['type'] != 'SCALAR' or index_doc.get('normalized'):
            raise FaceTextureError('Indices must be unsigned integer scalars')
        indices = accessor_array(doc, binary, primitive['indices']).astype(np.int64)
    else:
        indices = np.arange(len(positions))
    if len(indices) == 0 or len(indices) % 3 or indices.min() < 0 or indices.max() >= len(positions):
        raise FaceTextureError('Invalid triangle index table')
    triangles = indices.reshape(-1, 3)
    scenes, nodes = doc.get('scenes', []), doc.get('nodes', [])
    if len(scenes) != 1 or not nodes:
        raise FaceTextureError('Expected one explicit static scene')
    instances = []
    def visit(index, parent, ancestry):
        if index in ancestry:
            raise FaceTextureError('Cyclic node graph')
        node = nodes[index]
        if node.get('skin') is not None or node.get('weights') or node.get('extensions'):
            raise FaceTextureError('Skinned, weighted or extended nodes are unsupported')
        matrix = parent @ _node_matrix(node)
        if 'mesh' in node:
            if node['mesh'] != 0:
                raise FaceTextureError('Unexpected mesh index')
            instances.append(matrix)
        for child in node.get('children', []):
            visit(child, matrix, (*ancestry, index))
    if doc.get('scene', 0) != 0:
        raise FaceTextureError('Invalid default scene')
    for root in scenes[0].get('nodes', []):
        visit(root, np.eye(4), ())
    if len(instances) != 1:
        raise FaceTextureError('Expected exactly one mesh instance')
    world = positions @ instances[0][:3, :3].T + instances[0][:3, 3]
    try:
        material = doc['materials'][primitive['material']]
        color = material['pbrMetallicRoughness']['baseColorTexture']
        texture = doc['textures'][color['index']]
        image_index = texture['source']
    except (KeyError, IndexError) as exc:
        raise FaceTextureError('Source requires an embedded base-color texture and material') from exc
    if material.get('extensions') or color.get('extensions') or color.get('texCoord', 0) != 0 or texture.get('extensions') or material.get('alphaMode', 'OPAQUE') != 'OPAQUE':
        raise FaceTextureError('Extended/transformed textures, alternate UVs and non-opaque materials are unsupported')
    factor = np.asarray(material['pbrMetallicRoughness'].get('baseColorFactor', [1, 1, 1, 1]), dtype=float)
    if factor.shape != (4,) or not np.isfinite(factor).all() or np.any(factor[:3] <= 0):
        raise FaceTextureError('Base-color factor cannot encode the portrait')
    sampler = doc.get('samplers', [])[texture['sampler']] if 'sampler' in texture else {}
    with Image.open(io.BytesIO(images[image_index])) as image:
        atlas = np.array(image.convert('RGBA'))
    return {'raw': raw, 'gltf': doc, 'binary': binary, 'images': images,
            'attributes': attributes, 'positions': world, 'uv': uv, 'triangles': triangles,
            'primitive': primitive, 'material': material, 'texture': texture,
            'atlas': atlas, 'factor': factor, 'sampler': sampler}


def make_view(positions, resolution, yaw, bounds=None):
    angle = math.radians(yaw)
    rotation = np.array([[math.cos(angle), 0, -math.sin(angle)],
                         [0, 1, 0], [math.sin(angle), 0, math.cos(angle)]])
    camera = positions @ rotation.T
    if bounds is None:
        low, high = camera[:, :2].min(axis=0), camera[:, :2].max(axis=0)
        center = (low + high) / 2
        span = float(max(high - low) * 1.06)
        bounds = (center[0] - span/2, center[1] - span/2, center[0] + span/2, center[1] + span/2)
    x0, y0, x1, y1 = bounds
    if x1 <= x0 or y1 <= y0:
        raise FaceTextureError('Degenerate orthographic mesh bounds')
    xy = np.c_[(camera[:, 0] - x0) / (x1-x0) * resolution,
               (y1 - camera[:, 1]) / (y1-y0) * resolution]
    return {'xy': xy, 'depth': camera[:, 2], 'camera': camera,
            'bounds': list(bounds), 'rotation': rotation, 'resolution': resolution,
            'yaw_degrees': float(yaw)}


def _raster_triangle(triangle, width, height, centers):
    low = np.maximum(np.floor(triangle.min(axis=0) - centers).astype(int), 0)
    high = np.minimum(np.ceil(triangle.max(axis=0) - centers).astype(int), [width-1, height-1])
    if np.any(high < low):
        return None
    a, b, c = triangle
    e1, e2 = b-a, c-a
    determinant = e1[0]*e2[1] - e1[1]*e2[0]
    if abs(determinant) < 1e-10:
        return None
    xx, yy = np.meshgrid(np.arange(low[0], high[0]+1), np.arange(low[1], high[1]+1))
    points = np.c_[xx.ravel(), yy.ravel()]
    delta = points + centers - a
    u = (delta[:, 0]*e2[1] - delta[:, 1]*e2[0]) / determinant
    v = (e1[0]*delta[:, 1] - e1[1]*delta[:, 0]) / determinant
    bary = np.c_[1-u-v, u, v]
    inside = (bary >= -1e-7).all(axis=1)
    return points[inside], bary[inside]


def _texture_rgb(source, uv, atlas):
    height, width = atlas.shape[:2]
    coords = uv * [width, height] - .5
    # glTF's default wrapping is REPEAT. Clamp/mirror are handled per axis,
    # then bilinear neighbors use exactly the same wrapping at texture edges.
    indices = np.floor(coords).astype(int)
    fraction = coords - indices
    def wrapped(values, size, mode):
        if mode == 10497:
            return values % size
        if mode == 33071:
            return values.clip(0, size-1)
        if mode == 33648:
            values = values % (2*size)
            return np.where(values < size, values, 2*size-1-values)
        raise FaceTextureError('Unsupported texture wrap mode')
    x0 = wrapped(indices[:, 0], width, source['sampler'].get('wrapS', 10497))
    x1 = wrapped(indices[:, 0]+1, width, source['sampler'].get('wrapS', 10497))
    y0 = wrapped(indices[:, 1], height, source['sampler'].get('wrapT', 10497))
    y1 = wrapped(indices[:, 1]+1, height, source['sampler'].get('wrapT', 10497))
    fx, fy = fraction[:, 0:1], fraction[:, 1:2]
    rgb = (atlas[y0, x0, :3]*(1-fx)*(1-fy) + atlas[y0, x1, :3]*fx*(1-fy)
           + atlas[y1, x0, :3]*(1-fx)*fy + atlas[y1, x1, :3]*fx*fy)
    return rgb * source['factor'][:3]


def render_albedo(source, view, face_atlas=None, selected=None):
    resolution = view['resolution']
    rgb = np.full((resolution, resolution, 3), 245, dtype=np.uint8)
    depth = np.full((resolution, resolution), -np.inf)
    owners = np.full((resolution, resolution), -1, dtype=np.int32)
    xy = view['xy']
    projected = xy[source['triangles']]
    visible_box = ((projected.max(axis=1) >= 0).all(axis=1)
                   & (projected.min(axis=1) <= resolution).all(axis=1))
    for index in np.flatnonzero(visible_box):
        ids = source['triangles'][index]
        raster = _raster_triangle(xy[ids], resolution, resolution, .5)
        if raster is None:
            continue
        points, bary = raster
        if not len(points):
            continue
        z = bary @ view['depth'][ids]
        keep = z > depth[points[:, 1], points[:, 0]]
        points, bary, z = points[keep], bary[keep], z[keep]
        if not len(points):
            continue
        atlas = face_atlas if face_atlas is not None and selected[index] else source['atlas']
        values = _texture_rgb(source, bary @ source['uv'][ids], atlas)
        rgb[points[:, 1], points[:, 0]] = np.rint(values.clip(0, 255)).astype(np.uint8)
        depth[points[:, 1], points[:, 0]] = z
        owners[points[:, 1], points[:, 0]] = index
    return {'rgb': rgb, 'depth': depth, 'owners': owners, 'view': view}


def _resize(rgb, width, height, mask=False):
    return np.array(Image.fromarray(rgb).resize((max(1, int(width)), max(1, int(height))),
                    Image.Resampling.BILINEAR if mask else Image.Resampling.LANCZOS))


def _face_box(landmarks, width, height):
    xy = landmarks[:468, :2] * [width, height]
    low, high = xy.min(axis=0), xy.max(axis=0)
    center, size = (low+high)/2, max(high-low)*1.65
    box = (max(0, int(center[0]-size/2)), max(0, int(center[1]-size/2)),
           min(width, int(math.ceil(center[0]+size/2))), min(height, int(math.ceil(center[1]+size/2))))
    if min(box[2]-box[0], box[3]-box[1]) < 16:
        raise FaceTextureError('Detected face crop is too small')
    return box


def detect_face(detector, mp, rgb, label):
    """Search/refine RGB crops; every result is remapped to the actual image."""
    height, width = rgb.shape[:2]
    attempts = []
    def detect(box, kind, target=1024):
        x0, y0, x1, y1 = box
        patch = rgb[y0:y1, x0:x1]
        scale = target/max(patch.shape[:2])
        pixels = _resize(patch, round(patch.shape[1]*scale), round(patch.shape[0]*scale))
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(pixels)))
        attempts.append({'kind': kind, 'box_pixels': list(box), 'input_size': [pixels.shape[1], pixels.shape[0]], 'face_count': len(result.face_landmarks)})
        if len(result.face_landmarks) > 1:
            raise FaceTextureError(label + ': multiple faces are unsupported')
        found = []
        for raw in result.face_landmarks:
            lm = np.asarray([[p.x, p.y, p.z] for p in raw], dtype=float)
            if len(lm) < 468 or not np.isfinite(lm).all():
                raise FaceTextureError(label + ': invalid face landmarks')
            lm[:, 0] = (x0 + lm[:, 0]*(x1-x0))/width
            lm[:, 1] = (y0 + lm[:, 1]*(y1-y0))/height
            lm[:, 2] *= (x1-x0)/width
            found.append(lm)
        return found
    whole = (0, 0, width, height)
    found = detect(whole, 'full_image', max(width, height))
    if not found:
        found = detect(whole, 'upscaled_full_image', 1600)
    if not found:
        # Y-up source upper crops are tried before generic overlapping searches.
        unique = []
        boxes = [(width//4, 0, 3*width//4, height//2), (0, 0, width, height//2)]
        boxes += [(x, y, min(width, x+width//2), min(height, y+height//2))
                  for y in (0, height//4, height//2) for x in (0, width//4, width//2)]
        for box in boxes:
            for lm in detect(box, 'overlapping_search_crop'):
                span = np.linalg.norm((lm[263, :2]-lm[33, :2])*[width, height])
                duplicate = any(np.linalg.norm((lm[1, :2]-old[1, :2])*[width, height]) < .35*span for old in unique)
                if not duplicate:
                    unique.append(lm)
        found = unique
    if len(found) != 1:
        raise FaceTextureError(label + ': no unique usable face detected; use a matching frontal photo or correct source yaw')
    initial = found[0]
    box = _face_box(initial, width, height)
    refined = detect(box, 'detected_bbox_refinement')
    chosen = refined[0] if refined else initial
    span = float(np.linalg.norm((chosen[263, :2]-chosen[33, :2])*[width, height]))
    if span < 12:
        raise FaceTextureError(label + ': native outer-eye span is below 12 pixels')
    movement = np.linalg.norm((chosen[list(STABLE), :2]-initial[list(STABLE), :2])*[width, height], axis=1)
    if movement.max() > .4*span:
        raise FaceTextureError(label + ': crop refinement changed the face implausibly')
    if np.any(chosen[[33, 263, 152], :2] <= 0) or np.any(chosen[[33, 263, 152], :2] >= 1):
        raise FaceTextureError(label + ': registration anchors outside the image')
    eye = chosen[263, :2]-chosen[33, :2]
    lateral = abs(np.dot(chosen[1, :2]-(chosen[263, :2]+chosen[33, :2])/2, eye/np.linalg.norm(eye)))/np.linalg.norm(eye)
    if lateral > .4:
        raise FaceTextureError(label + ': face is too oblique for planar photo registration')
    return chosen, _face_box(chosen, width, height), {'face_count': 1, 'attempts': attempts,
            'eye_span_pixels': span, 'refinement_used': bool(refined),
            'refinement_max_pixels': float(movement.max()), 'nose_lateral_eye_fraction': float(lateral)}


def semantic_face_mask(segmenter, mp, rgb, landmarks, box, feather):
    height, width = rgb.shape[:2]
    def infer(pixels):
        result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(pixels)))
        masks = result.confidence_masks
        if masks is None or len(masks) != 6:
            raise FaceTextureError('Expected six selfie segmentation classes')
        skin = np.asarray(masks[2].numpy_view()).copy() + np.asarray(masks[3].numpy_view())
        hair = np.asarray(masks[1].numpy_view()).copy()
        if not np.isfinite(skin).all() or not np.isfinite(hair).all():
            raise FaceTextureError('Nonfinite semantic masks')
        return skin.astype(np.float32).clip(0, 1), hair.astype(np.float32).clip(0, 1)
    skin, hair = infer(rgb)
    skin, hair = _resize(skin, width, height, True), _resize(hair, width, height, True)
    x0, y0, x1, y1 = box
    patch = rgb[y0:y1, x0:x1]
    scale = 1024/max(patch.shape[:2])
    local_skin, local_hair = infer(_resize(patch, round(patch.shape[1]*scale), round(patch.shape[0]*scale)))
    skin[y0:y1, x0:x1] = _resize(local_skin, x1-x0, y1-y0, True)
    hair[y0:y1, x0:x1] = _resize(local_hair, x1-x0, y1-y0, True)
    oval = Image.new('L', (width, height))
    polygon = landmarks[list(FACE_OVAL), :2]*[width, height]
    ImageDraw.Draw(oval).polygon([tuple(p) for p in polygon], fill=255)
    inside = np.asarray(oval) > 0
    span = np.linalg.norm((landmarks[263, :2]-landmarks[33, :2])*[width, height])
    border = max(1, feather*span)
    oval_weight = np.clip(distance_transform_edt(inside)/border, 0, 1)
    semantic = np.clip((skin-.35)/.4, 0, 1) * np.clip((.35-hair)/.25, 0, 1)
    mask = (oval_weight*semantic).astype(np.float32)
    if (mask > .25).sum() < 16:
        raise FaceTextureError('Face semantics contain no usable skin region')
    return mask, {'skin_classes': [2, 3], 'hair_class': 1, 'face_oval_landmarks': list(FACE_OVAL),
                  'usable_face_pixels': int((mask > .25).sum()), 'feather_eye_fraction': float(feather)}


def register_faces(source_lm, photo_lm, source_shape, photo_shape):
    source_points = source_lm[:468, :2]*[source_shape[1], source_shape[0]]
    photo_points = photo_lm[:468, :2]*[photo_shape[1], photo_shape[0]]
    ids = [33, 263, 152]
    eye_span = float(np.linalg.norm(source_points[263]-source_points[33]))
    first, second = source_points[263]-source_points[33], source_points[152]-source_points[33]
    area = abs(float(first[0]*second[1]-first[1]*second[0]))
    if area/eye_span**2 < .25:
        raise FaceTextureError('Outer-eye/chin registration anchors are nearly collinear')
    affine = np.linalg.solve(np.c_[source_points[ids], np.ones(3)], photo_points[ids])
    singular = np.linalg.svd(affine[:2], compute_uv=False)
    if np.linalg.det(affine[:2]) <= 0 or singular[-1] <= 0 or singular[0]/singular[-1] > 2.5:
        raise FaceTextureError('Registration reverses orientation or stretches the face excessively')
    mapped = np.c_[source_points[list(STABLE)], np.ones(len(STABLE))] @ affine
    residual = np.linalg.norm(mapped-photo_points[list(STABLE)], axis=1)
    photo_span = float(np.linalg.norm(photo_points[263]-photo_points[33]))
    rms = float(np.sqrt(np.mean(residual**2)))
    if rms/photo_span > .3:
        raise FaceTextureError('Independent stable landmark residual exceeds 30% of photo eye span')
    return affine, {'kind': 'outer-eye/chin affine; source render pixels to original photo pixels',
                    'fit_landmark_indices': ids, 'diagnostic_landmark_indices': list(STABLE),
                    'affine_3x2': affine.tolist(), 'linear_singular_values': singular.tolist(),
                    'stable_residual_pixels': residual.tolist(), 'stable_rms_pixels': rms,
                    'stable_rms_photo_eye_fraction': rms/photo_span}


def _sample(mask, xy, order=1):
    return map_coordinates(mask, [xy[:, 1]-.5, xy[:, 0]-.5], order=order, mode='constant', cval=0)


def select_triangles(source, rendered, source_mask, photo_mask, affine):
    height, width = source_mask.shape
    yy, xx = np.nonzero((source_mask > .05) & (rendered['owners'] >= 0))
    screen = np.c_[xx+.5, yy+.5]
    photo_xy = np.c_[screen, np.ones(len(screen))] @ affine
    valid = _sample(photo_mask, photo_xy) > .05
    owners = rendered['owners'][yy[valid], xx[valid]]
    selected = np.zeros(len(source['triangles']), dtype=bool)
    selected[np.unique(owners)] = True
    # The mesh winding is retained. Visible back-facing hair sheets or rear
    # triangles never become portrait triangles, even for double-sided sources.
    camera = rendered['view']['camera'][source['triangles']]
    normal = np.cross(camera[:, 1]-camera[:, 0], camera[:, 2]-camera[:, 0])
    length = np.linalg.norm(normal, axis=1)
    selected &= normal[:, 2] > .05*length
    if not selected.any() or selected.all():
        raise FaceTextureError('No isolated front-facing face triangles could be selected')
    return selected


def paint_atlas(source, selected, rendered, source_mask, photo, photo_mask, affine, strength):
    atlas = source['atlas'].copy()
    original = source['atlas']
    height, width = atlas.shape[:2]
    covered = np.zeros((height, width), dtype=bool)
    strongest = np.zeros((height, width), dtype=np.float32)
    view = rendered['view']
    depth_tolerance = max(float(np.ptp(view['depth']))*1e-5, 1e-8)
    photo = photo.astype(np.float64)
    for face in np.flatnonzero(selected):
        ids = source['triangles'][face]
        # glTF normalized UVs address centers (i+.5)/N, not corners i/(N-1).
        tex = source['uv'][ids]*[width, height]-.5
        raster = _raster_triangle(tex, width, height, 0)
        if raster is None:
            continue
        points, bary = raster
        if not len(points):
            continue
        screen = bary @ view['xy'][ids]
        z = bary @ view['depth'][ids]
        px = np.floor(screen[:, 0]).astype(int)
        py = np.floor(screen[:, 1]).astype(int)
        in_frame = (px >= 0) & (py >= 0) & (px < source_mask.shape[1]) & (py < source_mask.shape[0])
        points, bary, screen, z, px, py = (a[in_frame] for a in (points, bary, screen, z, px, py))
        if not len(points):
            continue
        # Exact triangle ownership is strongest. Subpixel edge samples on an
        # adjacent triangle get a depth gate with the local triangle's slope.
        owner = rendered['owners'][py, px]
        sample_depth = rendered['depth'][py, px]
        xy_tri = view['xy'][ids]
        plane = np.linalg.lstsq(np.c_[xy_tri, np.ones(3)], view['depth'][ids], rcond=None)[0]
        local_tolerance = depth_tolerance + .75*(abs(plane[0])+abs(plane[1]))
        visible = (owner == face) | ((owner >= 0) & (np.abs(z-sample_depth) <= local_tolerance))
        photo_xy = np.c_[screen, np.ones(len(screen))] @ affine
        amount = _sample(source_mask, screen) * _sample(photo_mask, photo_xy) * strength
        keep = visible & (amount > 1e-5) & (amount > strongest[points[:, 1], points[:, 0]])
        points, amount, photo_xy = points[keep], amount[keep], photo_xy[keep]
        if not len(points):
            continue
        coords = [photo_xy[:, 1]-.5, photo_xy[:, 0]-.5]
        rgb = np.stack([map_coordinates(photo[:, :, channel], coords, order=1, mode='nearest') for channel in range(3)], axis=1)/source['factor'][:3]
        old = original[points[:, 1], points[:, 0], :3].astype(float)
        atlas[points[:, 1], points[:, 0], :3] = np.rint((old*(1-amount[:, None])+rgb*amount[:, None]).clip(0, 255)).astype(np.uint8)
        strongest[points[:, 1], points[:, 0]] = amount
        covered[points[:, 1], points[:, 0]] = True
    if not covered.any():
        raise FaceTextureError('Face projection covers no visible source atlas texels')
    distance, nearest = distance_transform_edt(~covered, return_indices=True)
    gutter = (distance > 0) & (distance <= 4)
    atlas[gutter] = atlas[nearest[0][gutter], nearest[1][gutter]]
    changed = np.any(atlas != original, axis=2)
    if not changed.any():
        raise FaceTextureError('Projection changed no texture pixels; no corrected output was produced')
    output = io.BytesIO()
    Image.fromarray(atlas).save(output, format='PNG')
    return output.getvalue(), {'edited_texture_pixels': int(changed.sum()),
                               'painted_texture_pixels': int(covered.sum()),
                               'gutter_pixels': int(gutter.sum()), 'gutter_width_pixels': 4,
                               'atlas_size': [width, height],
                               'scalar_base_color_factor': source['factor'].tolist()}


def partition_glb(source, selected, atlas_png):
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (len(source['triangles']),) or not selected.any() or selected.all():
        raise FaceTextureError('Expected a nonempty proper face triangle partition')
    original_doc = source['gltf']
    doc = copy.deepcopy(original_doc)
    binary = bytearray(source['binary'])
    def blob(data, target=None):
        binary.extend(b'\0'*(-len(binary) % 4))
        entry = {'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(data)}
        binary.extend(data)
        if target:
            entry['target'] = target
        doc['bufferViews'].append(entry)
        return len(doc['bufferViews'])-1
    def index_accessor(triangles):
        values = np.asarray(triangles, dtype='<u4').reshape(-1)
        doc['accessors'].append({'bufferView': blob(values.tobytes(), 34963),
                                'componentType': 5125, 'count': len(values), 'type': 'SCALAR'})
        return len(doc['accessors'])-1
    image_index = len(doc['images'])
    doc['images'].append({'name': 'Face-only photo atlas', 'mimeType': 'image/png', 'bufferView': blob(atlas_png)})
    material = copy.deepcopy(source['material'])
    texture = copy.deepcopy(source['texture'])
    texture['source'] = image_index
    texture_index = len(doc['textures'])
    doc['textures'].append(texture)
    material['name'] = 'Face texture only; unchanged original geometry'
    material['pbrMetallicRoughness']['baseColorTexture']['index'] = texture_index
    material_index = len(doc['materials'])
    doc['materials'].append(material)
    body, face = copy.deepcopy(source['primitive']), copy.deepcopy(source['primitive'])
    body['indices'] = index_accessor(source['triangles'][~selected])
    face['indices'] = index_accessor(source['triangles'][selected])
    face['material'] = material_index
    doc['meshes'][0]['primitives'] = [body, face]
    doc['buffers'][0]['byteLength'] = len(binary)
    encoded = json.dumps(doc, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' '*(-len(encoded) % 4)
    payload = bytes(binary)
    payload += b'\0'*(-len(payload) % 4)
    return (struct.pack('<4sII', b'glTF', 2, 28+len(encoded)+len(payload))
            + struct.pack('<II', len(encoded), 0x4E4F534A)+encoded
            + struct.pack('<II', len(payload), 0x004E4942)+payload)


def _comparison(photo, before, after, output):
    size = before.shape[0]
    reference = ImageOps.contain(Image.fromarray(photo), (size, size-32))
    canvas = Image.new('RGB', (size*3, size+32), (245, 245, 245))
    canvas.paste(reference, ((size-reference.width)//2, 32+(size-reference.height)//2))
    canvas.paste(Image.fromarray(before), (size, 32))
    canvas.paste(Image.fromarray(after), (size*2, 32))
    draw = ImageDraw.Draw(canvas)
    for index, title in enumerate(('Matching reference photo', 'Original mesh albedo', 'Face texture only / same geometry')):
        draw.text((index*size+10, 10), title, fill=(25, 25, 25))
    canvas.save(output, format='PNG')


def run_retexture(source_glb: Path, reference_image: Path, output_glb: Path,
                  comparison_png: Path, proof_json: Path, model_dir: Path,
                  strength: float = 1.0, feather: float = .18,
                  render_resolution: int = 768, yaw: float = 0.0) -> dict:
    """Observe, register, paint and append a private face texture; never deform."""
    if not np.isfinite([strength, feather, yaw]).all() or not 0 < strength <= 1 or not 0 <= feather <= 1 or not -180 <= yaw <= 180:
        raise FaceTextureError('Strength must be (0,1], feather [0,1], yaw [-180,180] degrees')
    if not isinstance(render_resolution, int) or not 256 <= render_resolution <= 2048:
        raise FaceTextureError('Render resolution must be an integer from 256 to 2048')
    source_glb, reference_image = Path(source_glb), Path(reference_image)
    output_glb, comparison_png, proof_json = Path(output_glb), Path(comparison_png), Path(proof_json)
    targets = [p.resolve() for p in (output_glb, comparison_png, proof_json)]
    if len(set(targets)) != 3 or any(p in (source_glb.resolve(), reference_image.resolve()) for p in targets):
        raise FaceTextureError('Output paths must be distinct and must not overwrite inputs')
    face_model = Path(model_dir)/'face_landmarker.task'
    segment_model = Path(model_dir)/'selfie_multiclass.tflite'
    for model in (face_model, segment_model):
        if not model.is_file():
            raise FaceTextureError('Missing local CPU model: '+str(model)+'; run install-worker.sh')
    source = read_source(source_glb)
    with Image.open(reference_image) as image:
        photo = np.array(ImageOps.exif_transpose(image).convert('RGB'))
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base = lambda path: python.BaseOptions(model_asset_path=str(path), delegate=python.BaseOptions.Delegate.CPU)
    options = vision.FaceLandmarkerOptions(base_options=base(face_model), running_mode=vision.RunningMode.IMAGE,
        num_faces=2, min_face_detection_confidence=.5, min_face_presence_confidence=.5, min_tracking_confidence=.5)
    full_view = make_view(source['positions'], render_resolution, yaw)
    full_render = render_albedo(source, full_view)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        photo_lm, photo_box, photo_observation = detect_face(detector, mp, photo, 'Reference photo')
        initial_lm, initial_box, full_observation = detect_face(detector, mp, full_render['rgb'], 'Source mesh render')
        # Re-render the detected head at native resolution rather than merely
        # enlarging low-resolution body pixels. Camera derives from observation.
        x0, y0, x1, y1 = initial_box
        left, bottom, right, top = full_view['bounds']
        bounds = (left+x0/render_resolution*(right-left), top-y1/render_resolution*(top-bottom),
                  left+x1/render_resolution*(right-left), top-y0/render_resolution*(top-bottom))
        # Clipping the detected crop at a full-render border must not change
        # the head's aspect ratio. Extend camera bounds, not source geometry.
        center_x, center_y = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
        half_span = max(bounds[2]-bounds[0], bounds[3]-bounds[1])/2
        bounds = (center_x-half_span, center_y-half_span, center_x+half_span, center_y+half_span)
        view = make_view(source['positions'], render_resolution, yaw, bounds)
        before = render_albedo(source, view)
        source_lm, source_box, source_observation = detect_face(detector, mp, before['rgb'], 'Refined source head render')
    segment_options = vision.ImageSegmenterOptions(base_options=base(segment_model), running_mode=vision.RunningMode.IMAGE,
                                                   output_confidence_masks=True, output_category_mask=False)
    with vision.ImageSegmenter.create_from_options(segment_options) as segmenter:
        source_mask, source_semantics = semantic_face_mask(segmenter, mp, before['rgb'], source_lm, source_box, feather)
        photo_mask, photo_semantics = semantic_face_mask(segmenter, mp, photo, photo_lm, photo_box, feather)
    affine, registration = register_faces(source_lm, photo_lm, before['rgb'].shape, photo.shape)
    selected = select_triangles(source, before, source_mask, photo_mask, affine)
    atlas_png, pixels = paint_atlas(source, selected, before, source_mask, photo, photo_mask, affine, strength)
    with Image.open(io.BytesIO(atlas_png)) as image:
        face_atlas = np.array(image.convert('RGBA'))
    after = render_albedo(source, view, face_atlas, selected)
    output = partition_glb(source, selected, atlas_png)
    doc, binary = parse_glb(output)
    original_doc = source['gltf']
    attributes_exact = all(p['attributes'] == source['primitive']['attributes'] for p in doc['meshes'][0]['primitives'])
    binary_exact = binary[:len(source['binary'])] == source['binary']
    indices_exact = all(np.array_equal(accessor_array(doc, binary, p['indices']).reshape(-1, 3), source['triangles'][mask])
                        for p, mask in zip(doc['meshes'][0]['primitives'], (~selected, selected)))
    original_images_exact = doc['images'][:len(original_doc['images'])] == original_doc['images'] and binary_exact
    transforms_exact = all(doc.get(key) == original_doc.get(key) for key in ('nodes', 'scenes', 'scene'))
    geometry_unchanged = binary_exact and attributes_exact and indices_exact and transforms_exact
    if not geometry_unchanged or not original_images_exact:
        raise FaceTextureError('Internal append-only export preservation check failed')
    changed_render = np.any(before['rgb'] != after['rgb'], axis=2)
    if not changed_render.any():
        raise FaceTextureError('The corrected texture has no visible effect in the actual source render')
    protected = (before['owners'] >= 0) & ~selected[np.maximum(before['owners'], 0)]
    protected_changes = int((changed_render & protected).sum())
    if protected_changes:
        raise FaceTextureError('Internal protected-body render isolation check failed')
    report = {'schema_version': 1, 'status': 'face_texture_only',
        'method': 'Fresh CPU MediaPipe RGB detection and semantics; orthographic visible-triangle photo projection; append-only private atlas',
        'source_sha256': hashlib.sha256(source['raw']).hexdigest(), 'output_sha256': hashlib.sha256(output).hexdigest(),
        'reference_sha256': hashlib.sha256(reference_image.read_bytes()).hexdigest(),
        'models': {'face_sha256': hashlib.sha256(face_model.read_bytes()).hexdigest(), 'segmenter_sha256': hashlib.sha256(segment_model.read_bytes()).hexdigest()},
        'original_vertex_rows': len(source['positions']), 'original_triangle_count': len(source['triangles']),
        'output_triangle_count': int((~selected).sum()+selected.sum()), 'selected_triangles': int(selected.sum()),
        'face_material_triangles': int(selected.sum()), 'original_material_triangles': int((~selected).sum()),
        'geometry_unchanged': bool(geometry_unchanged), 'original_geometry_binary_prefix_exact': binary_exact,
        'all_vertex_attributes_exact': attributes_exact, 'scene_transforms_exact': transforms_exact,
        'all_original_triangles_and_winding_retained': indices_exact, 'original_embedded_images_exact': original_images_exact,
        'original_binary_prefix_bytes': len(source['binary']), 'protected_body_changed_render_pixels': protected_changes,
        'changed_head_render_pixels': int(changed_render.sum()), 'generation': 'none', 'new_geometry': 0,
        'registration': registration, 'photo_observation': photo_observation,
        'source_full_observation': full_observation, 'source_observation': source_observation,
        'source_semantics': source_semantics, 'reference_semantics': photo_semantics,
        'camera': {'kind': 'orthographic albedo; Y-up; yaw=0 views from +Z', 'yaw_degrees': float(yaw),
                   'head_bounds_xy': view['bounds'], 'full_bounds_xy': full_view['bounds'], 'resolution': render_resolution},
        'strength': float(strength), **pixels,
        'limitations': ['Planar affine does not undo photographic perspective or expression differences.',
                        'Landmarks and semantic probabilities are observations, not anatomical reconstruction.',
                        'No vertex positions, UVs, normals, tangents or original image bytes are changed.']}
    for path in (output_glb, comparison_png, proof_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_glb.write_bytes(output)
    _comparison(photo, before['rgb'], after['rgb'], comparison_png)
    proof_json.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report
