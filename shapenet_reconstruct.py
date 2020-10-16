"""Reconstruct shape from point cloud using learned deformation space.
"""
import os
import sys
import argparse
import json
import trimesh
import torch
import numpy as np
import time
from types import SimpleNamespace

from shapenet_dataloader import ShapeNetMesh, FixedPointsCachedDataset
from shapeflow.layers.deformation_layer import NeuralFlowDeformer
from shapenet_embedding import LatentEmbedder


synset_to_cat = {
    "02691156": "airplane",
    "02933112": "cabinet",
    "03001627": "chair",
    "03636649": "lamp",
    "04090263": "rifle",
    "04379243": "table",
    "04530566": "watercraft",
    "02828884": "bench",
    "02958343": "car",
    "03211117": "display",
    "03691459": "speaker",
    "04256520": "sofa",
    "04401088": "telephone",
}

cat_to_synset = {value: key for key, value in synset_to_cat.items()}


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate reconstructions via retrieve and deform."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="path to input points (.ply file).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="path to output meshes."
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=4,
        help="top k nearest neighbor to retrieve.",
    )
    parser.add_argument(
        "-ne",
        "--embedding_niter",
        type=int,
        default=30,
        help="number of embedding iterations.",
    )
    parser.add_argument(
        "-nf",
        "--finetune_niter",
        type=int,
        default=30,
        help="number of finetuning iterations.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="path to pretrained checkpoint "
             "(params.json must be in the same directory).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device to run inference on.",
    )
    args = parser.parse_args()
    return args


def main():
    t0 = time.time()
    args_eval = get_args()

    device = torch.device(args_eval.device)

    # load training args
    run_dir = os.path.dirname(args_eval.checkpoint)
    args = SimpleNamespace(
        **json.load(open(os.path.join(run_dir, "params.json"), "r"))
    )

    # assert category is correct
    syn_id = args_eval.input_path.split("/")[-2]
    mesh_name = args_eval.input_path.split("/")[-1]
    assert syn_id == cat_to_synset[args.category]

    # output directories
    mesh_out_dir = os.path.join(args_eval.output_dir, "meshes", syn_id)
    mesh_out_file = os.path.join(
        mesh_out_dir, mesh_name.replace(".ply", ".off")
    )
    meta_out_dir = os.path.join(
        args_eval.output_dir, "meta", syn_id, mesh_name.replace(".ply", "")
    )
    orig_dir = os.path.join(meta_out_dir, "original_retrieved")
    deformed_dir = os.path.join(meta_out_dir, "deformed")
    os.makedirs(mesh_out_dir, exist_ok=True)
    os.makedirs(meta_out_dir, exist_ok=True)
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(deformed_dir, exist_ok=True)

    # redirect logging
    sys.stdout = open(os.path.join(meta_out_dir, "log.txt"), "w")

    # initialize deformer
    # input points
    points = np.array(trimesh.load(args_eval.input_path).vertices)

    # dataloader
    data_root = args.data_root
    mesh_dataset = ShapeNetMesh(
        data_root=data_root,
        split="train",
        category=args.category,
        normals=False,
    )
    point_dataset = FixedPointsCachedDataset(
        f"data/shapenet_pointcloud/train/{cat_to_synset[args.category]}.pkl",
        npts=300,
    )

    # setup model
    deformer = NeuralFlowDeformer(
        latent_size=args.lat_dims,
        f_width=args.deformer_nf,
        s_nlayers=2,
        s_width=5,
        method=args.solver,
        nonlinearity=args.nonlin,
        arch="imnet",
        adjoint=args.adjoint,
        rtol=args.rtol,
        atol=args.atol,
        via_hub=True,
        no_sign_net=(not args.sign_net),
        symm_dim=(2 if args.symm else None),
    )

    lat_params = torch.nn.Parameter(
        torch.randn(mesh_dataset.n_shapes, args.lat_dims) * 1e-1,
        requires_grad=True,
    )
    deformer.add_lat_params(lat_params)
    deformer.to(device)

    # load checkpoint
    resume_dict = torch.load(args_eval.checkpoint)
    deformer.load_state_dict(resume_dict["deformer_state_dict"])

    # embed
    embedder = LatentEmbedder(point_dataset, mesh_dataset, deformer, topk=5)
    input_pts = torch.tensor(points)[None].to(device)
    lat_codes_pre, lat_codes_post = embedder.embed(
        input_pts,
        matching="two_way",
        verbose=True,
        lr=1e-2,
        embedding_niter=args_eval.embedding_niter,
        finetune_niter=args_eval.finetune_niter,
        bs=4,
        seed=1,
    )

    # retrieve deformed models
    deformed_meshes, orig_meshes, dist = embedder.retrieve(
        lat_codes_post, tar_pts=points, matching="two_way"
    )
    asort = np.argsort(dist)
    dist = [dist[i] for i in asort]
    deformed_meshes = [deformed_meshes[i] for i in asort]
    orig_meshes = [orig_meshes[i] for i in asort]

    # output best mehs
    vb, fb = deformed_meshes[0]
    trimesh.Trimesh(vb, fb).export(mesh_out_file)

    # meta directory
    for i in range(len(deformed_meshes)):
        vo, fo = orig_meshes[i]
        vd, fd = deformed_meshes[i]
        trimesh.Trimesh(vo, fo).export(os.path.join(orig_dir, f"{i}.ply"))
        trimesh.Trimesh(vd, fd).export(os.path.join(deformed_dir, f"{i}.ply"))
    np.save(os.path.join(meta_out_dir, "latent.npy"), lat_codes_pre)
    t1 = time.time()
    print(f"Total Timelapse: {t1-t0:.4f}")


if __name__ == "__main__":
    main()
