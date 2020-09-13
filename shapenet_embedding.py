import numpy as np
from scipy.spatial import cKDTree
import time
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader

from shapeflow.layers.chamfer_layer import ChamferDistKDTree
from shapeflow.layers.shared_definition import LOSSES, OPTIMIZERS
import shapeflow.utils.train_utils as utils


class LatentEmbedder(object):
    """Helper class for embedding new observation in deformation latent space.
    """

    def __init__(self, point_dataset, mesh_dataset, deformer, topk=5):
        """Initialize embedder.

        Args:
          point_dataset: instance of FixedPointsCachedDataset
          mesh_dataset: instance of ShapeNetMesh
          deformer: pretrined deformer instance
        """
        self.point_dataset = point_dataset
        self.mesh_dataset = mesh_dataset
        self.deformer = deformer
        self.topk = topk
        self.tree = cKDTree(self.lat_params.clone().detach().cpu().numpy())

    @property
    def lat_dims(self):
        return self.lat_params.shape[1]

    @property
    def lat_params(self):
        return self.deformer.net.lat_params

    @property
    def symm(self):
        return self.deformer.symm_dim is not None

    @property
    def device(self):
        return self.lat_params.device

    def _padded_verts_from_meshes(self, meshes):
        verts = [vf[0] for vf in meshes]
        faces = [vf[1] for vf in meshes]
        nv = [v.shape[0] for v in verts]
        max_nv = np.max(nv)
        verts_pad = [
            np.pad(verts[i], ((0, max_nv - nv[i]), (0, 0)))
            for i in range(len(nv))
        ]
        verts_pad = np.stack(verts_pad, 0)  # [nmesh, max_nv, 3]
        return verts_pad, faces, nv

    def _meshes_from_padded_verts(self, verts_pad, faces, nv):
        verts_pad = [v for v in verts_pad]
        verts = [v[:n] for v, n in zip(verts_pad, nv)]
        meshes = list(zip(verts, faces))
        return meshes

    def embed(
        self,
        input_points,
        optimizer="adam",
        lr=1e-3,
        seed=0,
        embedding_niter=30,
        finetune_niter=30,
        bs=32,
        verbose=False,
        matching="two_way",
        loss_type="l1",
    ):
        """Embed inputs points observations into deformation latent space.

        Args:
          input_points: tensor of shape [bs_tar, npoints, 3]
          optimizer: str, optimizer choice. one of sgd, adam, adadelta,
            adagrad, rmsprop.
          lr: float, learning rate.
          seed: int, random seed.
          embedding_niter: int, number of embedding optimization iterations.
          finetune_niter: int, number of finetuning optimization iterations.
          bs: int, batch size.
          verbose: bool, turn on verbose.
          matching: str, matching function. choice of one_way or two_way.
          loss_type: str, loss type. choice of l1, l2, huber.

        Returns:
          embedded_latents: tensor of shape [batch, lat_dims]
        """
        if input_points.shape[0] != 1:
            raise NotImplementedError("Code is not ready for batch size > 1.")
        torch.manual_seed(seed)

        # Check input validity.
        if matching not in ["one_way", "two_way"]:
            raise ValueError(
                f"matching method must be one of one_way / two_way. Instead "
                f"entered {matching}"
            )
        if loss_type not in LOSSES.keys():
            raise ValueError(
                f"loss_type must be one of {LOSSES.keys()}. "
                f"Instead entered {loss_type}"
            )

        criterion = LOSSES[loss_type]

        bs_tar, npts_tar, _ = input_points.shape
        # Assign random latent code close to zero.
        embedded_latents = torch.nn.Parameter(
            torch.randn(bs_tar, self.lat_dims, device=self.device) * 1e-4,
            requires_grad=True,
        )
        self.deformer.net.tar_latents = embedded_latents
        embedded_latents = self.deformer.net.tar_latents
        # [bs_tar, lat_dims]

        # Init optimizer.
        if optimizer not in OPTIMIZERS.keys():
            raise ValueError(f"optimizer must be one of {OPTIMIZERS.keys()}")
        optim = OPTIMIZERS[optimizer]([embedded_latents], lr=lr)

        # Init dataloader.
        sampler = SubsetRandomSampler(
            np.arange(len(self.point_dataset)).tolist()
        )
        point_loader = DataLoader(
            self.point_dataset,
            batch_size=bs,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
        )

        # Chamfer distance calc.
        chamfer_dist = ChamferDistKDTree(reduction="mean", njobs=1)
        chamfer_dist.to(self.device)

        def optimize_latent(point_loader, optim, niter):
            # Optimize for latents.
            self.deformer.train()
            toc = time.time()

            bs_src = point_loader.batch_size
            embedded_latents_ = embedded_latents[None].expand(
                bs_src, bs_tar, self.lat_dims
            )
            # [bs_src, bs_tar, lat_dims]

            # Broadcast and reshape input points.
            target_points_ = (
                input_points[None]
                .expand(bs_src, bs_tar, npts_tar, 3)
                .view(-1, npts_tar, 3)
            )
            it = 0

            for batch_idx, (fnames, idxs, source_points) in enumerate(
                point_loader
            ):
                tic = time.time()
                # Send tensors to device.
                source_points = source_points.to(
                    self.device
                )  # [bs_src, npts_src, 3]
                idxs = idxs.to(self.device)

                optim.zero_grad()

                # Deform chosen points to input_points.
                # Broadcast src lats to src x tar.
                source_latents = self.lat_params[idxs]  # [bs_src, lat_dims]
                source_latents_ = source_latents[:, None].expand(
                    bs_src, bs_tar, self.lat_dims
                )
                source_latents_ = source_latents_.view(-1, self.lat_dims)
                target_latents_ = embedded_latents_.view(-1, self.lat_dims)
                zeros = torch.zeros_like(source_latents_)
                source_target_latents = torch.stack(
                    [source_latents_, zeros, target_latents_], dim=1
                )

                deformed_pts = self.deformer(
                    source_points,
                    source_target_latents,  # [bs_sr*bs_tar, npts_src, 3]
                )  # [bs_sr*bs_tar, npts_src, 3]

                # Symmetric pair of matching losses.
                if self.symm:
                    accu, comp, cham = chamfer_dist(
                        utils.symmetric_duplication(deformed_pts, symm_dim=2),
                        utils.symmetric_duplication(
                            target_points_, symm_dim=2
                        ),
                    )
                else:
                    accu, comp, cham = chamfer_dist(
                        deformed_pts, target_points_
                    )

                if matching == "one_way":
                    comp = torch.mean(comp, dim=1)
                    loss = criterion(comp, torch.zeros_like(comp))
                else:
                    loss = criterion(cham, torch.zeros_like(cham))

                # Check amount of deformation.
                deform_abs = torch.mean(
                    torch.norm(deformed_pts - source_points, dim=-1)
                )

                loss.backward()

                # Gradient clipping.
                torch.nn.utils.clip_grad_value_(embedded_latents, 1.0)

                optim.step()

                toc = time.time()
                if verbose:
                    if loss_type == "l1":
                        dist = loss.item()
                    else:
                        dist = np.sqrt(loss.item())
                    print(
                        f"Iter: {it}, Loss: {loss.item():.4f}, "
                        f"Dist: {dist:.4f}, "
                        f"Deformation Magnitude: {deform_abs.item():.4f}, "
                        f"Time per iter (s): {toc-tic:.4f}"
                    )
                    it += 1
                if batch_idx >= niter:
                    break

        # Optimize to range.
        optimize_latent(point_loader, optim, embedding_niter)
        latents_pre_tune = embedded_latents.detach().cpu().numpy()

        # Finetune topk.
        dist, idxs = self.tree.query(
            embedded_latents.detach().cpu().numpy(), k=self.topk
        )  # [batch, k]
        bs, k = idxs.shape
        idxs_ = idxs.reshape(-1)

        # Change lr.
        for param_group in optim.param_groups:
            param_group["lr"] = 1e-3

        sampler = SubsetRandomSampler(idxs_.tolist() * finetune_niter)
        point_loader = DataLoader(
            self.point_dataset,
            batch_size=self.topk,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
        )

        print(f"Finetuning for {finetune_niter} iters...")
        optim = OPTIMIZERS[optimizer](
            [embedded_latents] + list(self.deformer.parameters()), lr=1e-3
        )
        optimize_latent(point_loader, optim, finetune_niter)
        latents_post_tune = embedded_latents.detach().cpu().numpy()

        return latents_pre_tune, latents_post_tune

    def retrieve(self, lat_codes, tar_pts, matching="one_way"):
        """Retrieve top 10 nearest neighbors, deform and pick the best one.

        Args:
          lat_codes: tensor of shape [batch, lat_dims], latent code targets.

        Returns:
          List of len batch of (V, F) tuples.
        """
        if lat_codes.shape[0] != 1:
            raise NotImplementedError("Code is not ready for batch size > 1.")
        dist, idxs = self.tree.query(lat_codes, k=self.topk)  # [batch, k]
        bs, k = idxs.shape
        idxs_ = idxs.reshape(-1)

        if not isinstance(lat_codes, torch.Tensor):
            lat_codes = torch.tensor(lat_codes).float().to(self.device)

        src_latent = self.lat_params[idxs_]  # [batch*k, lat_dims]
        tar_latent = (
            lat_codes[:, None]
            .expand(bs, k, self.lat_dims)
            .reshape(-1, self.lat_dims)
        )  # [batch*k, lat_dims]
        zeros = torch.zeros_like(src_latent)
        src_tar_latent = torch.stack([src_latent, zeros, tar_latent], dim=1)

        # Retrieve meshes.
        orig_meshes = [
            self.mesh_dataset.get_single(i) for i in idxs_
        ]  # [(v1,f1), ..., (vn,fn)]
        src_verts, faces, nv = self._padded_verts_from_meshes(orig_meshes)
        src_verts = torch.tensor(src_verts).to(self.device)
        with torch.no_grad():
            deformed_verts = self.deformer(src_verts, src_tar_latent)
        deformed_meshes = self._meshes_from_padded_verts(
            deformed_verts, faces, nv
        )

        # Chamfer distance calc.
        chamfer_dist = ChamferDistKDTree(reduction="mean", njobs=1)
        chamfer_dist.to(self.device)
        dist = []
        for i in range(len(deformed_meshes)):
            accu, comp, cham = chamfer_dist(
                deformed_meshes[i][0][None].to(self.device),
                torch.tensor(tar_pts)[None].to(self.device),
            )
            if matching == "one_way":
                dist.append(torch.mean(comp, dim=1).item())
            else:
                dist.append(cham.item())

        # Reshape the list of (v, f) tuples.
        deformed_meshes = [
            (vf[0].detach().cpu().numpy(), vf[1].detach().cpu().numpy())
            for vf in deformed_meshes
        ]

        return deformed_meshes, orig_meshes, dist
