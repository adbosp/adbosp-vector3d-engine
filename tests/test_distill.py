"""Level-1 distillation: procedural teachers and training samples (no torch needed)."""
import numpy as np

from vector3d.lab import dataset as ds
from vector3d.lab import distill


def test_random_objects_are_valid():
    for i in range(6):
        mesh = ds.mesh_sdf(distill.random_sdf(np.random.default_rng(i)), 48)
        assert len(mesh.faces) > 100


def test_mesh_occupancy_of_sphere():
    mesh = ds.mesh_sdf(lambda p: ds.sd_sphere(p, 0.5), 48)
    occ = distill.mesh_occupancy(mesh, (0, 0, 0), 0.75, 32)
    vol = occ.mean() * 1.5 ** 3
    assert abs(vol - 4 / 3 * np.pi * 0.125) / (4 / 3 * np.pi * 0.125) < 0.3      # boundary voxels count as full at this resolution


def test_make_sample_shapes():
    mesh = ds.mesh_sdf(distill.random_sdf(np.random.default_rng(1)), 48)
    s = distill.make_sample(mesh, size=64, G=32)
    assert s["x"].shape == (len(distill.CHANNELS), 32, 32, 32)
    y = np.unpackbits(s["y"])[: 32 ** 3].reshape(32, 32, 32)
    hull = s["x"][0] > 0.5
    assert y.any() and (y.astype(bool) & ~hull).mean() < 0.03      # inside its own hull, up to boundary voxels
