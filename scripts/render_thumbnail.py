"""Render thumbnails for meshes.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.render import render_mesh
import scipy.misc
import tqdm
import argparse
import glob
import imageio


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Render thumbnails.")
    
    parser.add_argument("--file_pattern", type=str, default="*/*/*/*.ply",
                        help="filename pattern for files to be rendered.")
    parser.add_argument("--input_root",  type=str, default="data/shapenet_simplified",
                        help="path to input mesh root.")
    parser.add_argument("--output_root", type=str, default="data/shapenet_thumbnails",
                        help="path to output image root.")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    in_files = glob.glob(os.path.join(args.input_root, args.file_pattern))
    for in_file in tqdm.tqdm(in_files):
        out_file = in_file.replace(args.input_root, args.output_root)
        out_dir = os.path.dirname(out_file)
        os.makedirs(out_dir, exist_ok=True)
        image, _ = render_mesh(in_file, dist=1.2, res=(112, 112), color=[0.,1.,0.], intensity=6)
        imageio.imwrite(os.path.join(out_dir, 'thumbnail.jpg'), image)
        
main()