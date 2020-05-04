"""Render thumbnails for meshes.
"""

import os
import sys
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(WORKING_DIR)
from utils.render import render_mesh
from multiprocessing import Pool
import scipy.misc
import tqdm
import argparse
import glob
import imageio


def render_file(in_file):
    out_file = in_file.replace(input_root, output_root)
    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)
    image, _ = render_mesh(in_file, dist=1.2, res=(112, 112), color=[0.,1.,0.], intensity=6)
    imageio.imwrite(os.path.join(out_dir, 'thumbnail.jpg'), image)


def wrapper(arg):
    return arg, render_file(arg)


def update(args):
    pbar.update()


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Render thumbnails.")
    
    parser.add_argument("--file_pattern", type=str, default="**/*.ply",
                        help="filename pattern for files to be rendered.")
    parser.add_argument("--input_root",  type=str, default="data/shapenet_simplified",
                        help="path to input mesh root.")
    parser.add_argument("--output_root", type=str, default="data/shapenet_thumbnails",
                        help="path to output image root.")
    parser.add_argument("--n_jobs", type=int, default=-1, 
                        help="Number of processes to use. Use all if set to -1.")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    in_files = glob.glob(os.path.join(WORKING_DIR, args.input_root, args.file_pattern), recursive=True)
    global input_root
    global output_root
    global pbar
    input_root = args.input_root
    output_root = args.output_root
    pbar = tqdm.tqdm(total=len(in_files))
    pool = Pool(processes=None if args.n_jobs == -1 else args.n_jobs)
    for fname in in_files:
        pool.apply_async(wrapper, args=(fname,), callback=update)
    pool.close()
    pool.join()
    pbar.close()        
        
main()