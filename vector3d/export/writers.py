"""Mesh writers: OBJ and binary glTF (GLB). World frame already matches glTF (+Y up, +Z front)."""
from __future__ import annotations

import io
import json
import struct

import numpy as np

from ..reconstruct.surface import Mesh


def to_obj(mesh: Mesh, name: str = "vector3d") -> str:
    out = io.StringIO()
    out.write(f"# Vector3D Engine\no {name}\n")
    np.savetxt(out, mesh.vertices, fmt="v %.6f %.6f %.6f")
    np.savetxt(out, mesh.normals(), fmt="vn %.6f %.6f %.6f")
    f = mesh.faces + 1
    np.savetxt(out, np.column_stack([f[:, 0], f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2]]), fmt="f %d//%d %d//%d %d//%d")
    return out.getvalue()


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * (-len(b) % 4)


def to_glb(mesh: Mesh, name: str = "vector3d", extras: dict | None = None) -> bytes:
    pos = mesh.vertices.astype(np.float32)
    nrm = mesh.normals().astype(np.float32)
    idx = mesh.faces.astype(np.uint32).ravel()
    blobs = [pos.tobytes(), nrm.tobytes(), idx.tobytes()]
    offsets, buf = [], b""
    for b in blobs:
        offsets.append(len(buf))
        buf += _pad4(b)
    gltf = {
        "asset": {"version": "2.0", "generator": "Vector3D Engine"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"name": "default", "pbrMetallicRoughness": {"baseColorFactor": [0.75, 0.77, 0.8, 1.0],
                                                                    "metallicFactor": 0.0, "roughnessFactor": 0.6},
                       "doubleSided": False}],
        "buffers": [{"byteLength": len(buf)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(blobs[0]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(blobs[1]), "target": 34962},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(blobs[2]), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(nrm), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
        ],
    }
    if extras:
        gltf["extras"] = extras
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    total = 12 + 8 + len(js) + 8 + len(buf)
    return (struct.pack("<4sII", b"glTF", 2, total)
            + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(buf), b"BIN\x00") + buf)
