# load libraries
import trimesh
import torch
import json
import os
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from utils import render
from shapenet_dataloader import ShapeNetMesh
from deepdeform.layers.deformation_layer import NeuralFlowDeformer

synset_to_cat = {
    '02691156': 'airplane',
    '02933112': 'cabinet',
    '03001627': 'chair',
    '03636649': 'lamp',
    '04090263': 'rifle',
    '04379243': 'table',
    '04530566': 'watercraft',
    '02828884': 'bench',
    '02958343': 'car',
    '03211117': 'display',
    '03691459': 'speaker',
    '04256520': 'sofa',
    '04401088': 'telephone'
}

cat_to_synset = {value:key for key, value in synset_to_cat.items()}

############## CONFIG (UPDATE BEFORE EACH RUN) #############
# choice of checkpoint to load
run_dir =  "runs/pretrained_ckpt"
checkpoint = "checkpoint_latest.pth.tar_deepdeform_100.pth.tar"  # "checkpoint_latest.pth.tar_deepdeform_034.pth.tar" #  "checkpoint_latest.pth.tar_deepdeform_100.pth.tar"
device = torch.device("cuda:0")
out_dir = "out"
start_id = 0
end_id = 1582 # 4746

################### SETUP ###################
# setup
# load training args
args = SimpleNamespace(**json.load(open(os.path.join(run_dir, 'params.json'), 'r')))
if not 'symm' in args.__dict__.keys():
    args.symm = None
if not 'category' in args.__dict__.keys():
    args.category = "chair"

# dataloader
data_root = args.data_root
dset = ShapeNetMesh(data_root=data_root, split="train", category="chair", normals=False)

# output dir
can_dir = os.path.join(out_dir, f"canonicalized_mesh/{cat_to_synset[args.category]}")
orig_dir = os.path.join(out_dir, f"original_mesh/{cat_to_synset[args.category]}")
os.makedirs(can_dir, exist_ok=True)
os.makedirs(orig_dir, exist_ok=True)

# setup model
deformer = NeuralFlowDeformer(latent_size=args.lat_dims, f_width=args.deformer_nf, s_nlayers=2, 
                              s_width=5, method=args.solver, nonlinearity=args.nonlin, arch='imnet',
                              adjoint=args.adjoint, rtol=args.rtol, atol=args.atol, via_hub=True,
                              no_sign_net=(not args.sign_net), symm_dim=(2 if args.symm else None))
lat_params = torch.nn.Parameter(torch.randn(dset.n_shapes, args.lat_dims)*1e-1, requires_grad=True)
deformer.add_lat_params(lat_params)
deformer.to(device)

# load checkpoint
resume_dict = torch.load(os.path.join(run_dir, checkpoint))
start_ep = resume_dict["epoch"]
global_step = resume_dict["global_step"]
tracked_stats = resume_dict["tracked_stats"]
deformer.load_state_dict(resume_dict["deformer_state_dict"])

################### EVAL ###################
lat_path = lambda l_src_, l_tar_: torch.stack([l_src_, l_tar_], dim=1)
ten2npy = lambda tensor: tensor.detach().cpu().numpy()
    
for idx in tqdm(range(start_id, end_id)):
    # get source mesh
    v, f = dset.get_single(idx)
    v_ = v[None].to(device)
    f_ = f.detach().cpu().numpy()

    with torch.no_grad():
        # get the latent codes corresponding to these shapes
        l_src = deformer.get_lat_params(idx)[None] 
        l_tar = torch.zeros_like(l_src)  # target is the "hub"

        # deform source to target
        v_s2t = deformer(v_, lat_path(l_src, l_tar))[0]  # source to target
        mesh_d = trimesh.Trimesh(ten2npy(v_s2t), f_, process=False)
        mesh_o = trimesh.Trimesh(ten2npy(v_[0]), f_, process=False)
        fname = dset.fnames[idx].split('/')[-1]
        mesh_d.export(os.path.join(can_dir, fname+".ply"))
        mesh_o.export(os.path.join(orig_dir, fname+".ply"))