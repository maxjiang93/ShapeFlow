"""Precompute and cache shapenet pointclouds.
"""

import os
import sys

WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(WORKING_DIR)
import tqdm  # noqa: E402
import argparse  # noqa: E402
import glob  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from multiprocessing import Pool  # noqa: E402
import pickle  # noqa: E402
from collections import OrderedDict  # noqa: E402


def sample_vertex(filename):
    mesh = trimesh.load(filename)
    v = np.array(mesh.vertices, dtype=np.float32)
    np.random.shuffle(v)
    return v[:n_points]


def wrapper(arg):
    return arg, sample_vertex(arg)


def update(args):
    filename, point_samp = args
    fname = "/".join(filename.split("/")[-4:-1])
    # note: input comes from async `wrapper`
    sampled_points[
        fname
    ] = point_samp  # put answer into correct index of result list
    pbar.update()


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute and cache shapenet pointclouds."
    )

    parser.add_argument(
        "--file_pattern",
        type=str,
        default="**/*.ply",
        help="filename pattern for files to be rendered.",
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="data/shapenet_simplified",
        help="path to input mesh root.",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        default="data/shapenet_points.pkl",
        help="path to output image root.",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=4096,
        help="Number of points to sample per shape",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of processes to use. Use all if set to -1.",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    patt = os.path.join(WORKING_DIR, args.input_root, args.file_pattern)
    in_files = glob.glob(patt, recursive=True)
    global sampled_points
    global n_points
    global pbar
    sampled_points = OrderedDict()
    n_points = args.n_points
    pbar = tqdm.tqdm(total=len(in_files))
    pool = Pool(processes=None if args.n_jobs == -1 else args.n_jobs)
    for fname in in_files:
        pool.apply_async(wrapper, args=(fname,), callback=update)
    pool.close()
    pool.join()
    pbar.close()
    with open(args.output_pkl, "wb") as fh:
        pickle.dump(sampled_points, fh)


if __name__ == "__main__":
    main()
