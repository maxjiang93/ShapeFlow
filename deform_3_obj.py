import torch.optim as optim

from shapeflow.layers.chamfer_layer import ChamferDistKDTree
from shapeflow.layers.deformation_layer import NeuralFlowDeformer
from shapeflow.layers.pointnet_layer import PointNetEncoder

import torch
import numpy as np
from time import time
import trimesh
from glob import glob


files = sorted(glob("data/shapenet_watertight/val/03001627/*/*.ply"))
m1 = trimesh.load(files[1])
m2 = trimesh.load(files[6])
m3 = trimesh.load(files[7])
device = torch.device("cuda:0")

chamfer_dist = ChamferDistKDTree(reduction="mean", njobs=1).to(device)
criterion = torch.nn.MSELoss()

latent_size = 3

deformer = NeuralFlowDeformer(
    latent_size=latent_size,
    f_nlayers=6,
    f_width=100,
    s_nlayers=2,
    s_width=5,
    method="dopri5",
    nonlinearity="elu",
    arch="imnet",
    adjoint=True,
    atol=1e-4,
    rtol=1e-4,
).to(device)
encoder = PointNetEncoder(
    nf=16, out_features=latent_size, dropout_prob=0.0
).to(device)

# this is an awkward workaround to get gradients for encoder via adjoint solver
deformer.add_encoder(encoder)
deformer.to(device)
encoder = deformer.net.encoder

optimizer = optim.Adam(list(deformer.parameters()), lr=1e-3)

niter = 1000
npts = 5000

V1 = torch.tensor(m1.vertices.astype(np.float32)).to(device)  # .unsqueeze(0)
V2 = torch.tensor(m2.vertices.astype(np.float32)).to(device)  # .unsqueeze(0)
V3 = torch.tensor(m3.vertices.astype(np.float32)).to(device)  # .unsqueeze(0)

loss_min = 1e30
tic = time()
encoder.train()

for it in range(0, niter):
    optimizer.zero_grad()

    seq1 = torch.randperm(V1.shape[0], device=device)[:npts]
    seq2 = torch.randperm(V2.shape[0], device=device)[:npts]
    seq3 = torch.randperm(V3.shape[0], device=device)[:npts]
    V1_samp = V1[seq1]
    V2_samp = V2[seq2]
    V3_samp = V3[seq3]

    V_src = torch.stack(
        [V1_samp, V1_samp, V2_samp], dim=0
    )  # [batch, npoints, 3]
    V_tar = torch.stack(
        [V2_samp, V3_samp, V3_samp], dim=0
    )  # [batch, npoints, 3]

    V_src_tar = torch.cat([V_src, V_tar], dim=0)
    V_tar_src = torch.cat([V_tar, V_src], dim=0)

    batch_latent_src_tar = encoder(V_src_tar)
    batch_latent_tar_src = torch.cat(
        [batch_latent_src_tar[3:], batch_latent_src_tar[:3]]
    )

    V_deform = deformer(V_src_tar, batch_latent_src_tar, batch_latent_tar_src)

    _, _, dist = chamfer_dist(V_deform, V_tar_src)

    loss = criterion(dist, torch.zeros_like(dist))

    loss.backward()
    optimizer.step()

    if it % 100 == 0 or True:
        print(f"iter={it}, loss={np.sqrt(loss.item())}")

toc = time()
print("Time for {} iters: {:.4f} s".format(niter, toc - tic))

# save deformed mesh
encoder.eval()
with torch.no_grad():
    V1_latent = encoder(V1.unsqueeze(0))
    V2_latent = encoder(V2.unsqueeze(0))
    V3_latent = encoder(V3.unsqueeze(0))

    V1_2 = (
        deformer(V1.unsqueeze(0), V1_latent, V2_latent)
        .detach()
        .cpu()
        .numpy()[0]
    )
    V2_1 = (
        deformer(V2.unsqueeze(0), V2_latent, V1_latent)
        .detach()
        .cpu()
        .numpy()[0]
    )
    V1_3 = (
        deformer(V1.unsqueeze(0), V1_latent, V3_latent)
        .detach()
        .cpu()
        .numpy()[0]
    )
    V3_1 = (
        deformer(V3.unsqueeze(0), V3_latent, V1_latent)
        .detach()
        .cpu()
        .numpy()[0]
    )
    V2_3 = (
        deformer(V2.unsqueeze(0), V2_latent, V3_latent)
        .detach()
        .cpu()
        .numpy()[0]
    )
    V3_2 = (
        deformer(V3.unsqueeze(0), V3_latent, V2_latent)
        .detach()
        .cpu()
        .numpy()[0]
    )
trimesh.Trimesh(V1_2, m1.faces).export("demo/output_1_2.obj")
trimesh.Trimesh(V2_1, m2.faces).export("demo/output_2_1.obj")
trimesh.Trimesh(V1_3, m1.faces).export("demo/output_1_3.obj")
trimesh.Trimesh(V3_1, m3.faces).export("demo/output_3_1.obj")
trimesh.Trimesh(V2_3, m2.faces).export("demo/output_2_3.obj")
trimesh.Trimesh(V3_2, m3.faces).export("demo/output_3_2.obj")

m1.export("demo/output_1.obj")
m2.export("demo/output_2.obj")
m3.export("demo/output_3.obj")
