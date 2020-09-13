"""Script for performing mesh simplification.
"""
# flake8: noqa E402
import sys

sys.path.append("..")

import meshutils
import trimesh
import glob
import os
import tqdm


files = glob.glob("../data/shapenet_watertight/test/*/*/*.ply")
outroot = "shapenet_simplified"

for f_in in tqdm.tqdm(files):
    f_out = f_in.replace("shapenet_watertight", outroot)
    dirname = os.path.dirname(f_out)
    os.makedirs(dirname, exist_ok=True)
    mesh = trimesh.load(f_in)
    v, f = meshutils.fast_simplify(mesh.vertices, mesh.faces, ratio=0.1)
    trimesh.Trimesh(v, f).export(f_out)
