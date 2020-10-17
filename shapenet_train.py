"""Training script shapenet deformation space experiment.
"""
import argparse
import json
import os
import glob
import numpy as np
import time
import trimesh

import shapeflow.utils.train_utils as utils
from shapeflow.layers.chamfer_layer import ChamferDistKDTree
from shapeflow.layers.deformation_layer import NeuralFlowDeformer
import shapenet_dataloader as dl

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(precision=4)


# Various choices for losses and optimizers.
LOSSES = {
    "l1": torch.nn.L1Loss(),
    "l2": torch.nn.MSELoss(),
    "huber": torch.nn.SmoothL1Loss(),
}

OPTIMIZERS = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "rmsprop": optim.RMSprop,
}

SOLVERS = [
    "dopri5",
    "adams",
    "euler",
    "midpoint",
    "rk4",
    "explicit_adams",
    "fixed_adams",
    "bosh3",
    "adaptive_heun",
    "tsit5",
]


def compute_latent_dict(deformer, dataset):
    """
    Args:
        deformer:
        dataset:
    Returns:
        a dict that maps filenames to latent codes.
    """
    # Encode all shapes from dataloader into latents.
    all_filenames = dataset.file_splits["train"]
    all_filenames = [dl.strip_name(f) for f in all_filenames]
    if isinstance(deformer, nn.DataParallel):
        all_latents = deformer.module.net.lat_params.detach().cpu().numpy()
    else:
        all_latents = deformer.net.lat_params.detach().cpu().numpy()

    return dict(zip(all_filenames, all_latents))


def get_k(epoch):
    if epoch < 10:
        return 4000
    elif epoch < 50:
        return 800
    elif epoch < 80:
        return 100
    else:
        return 10


def train_or_eval(
    mode,
    args,
    deformer,
    chamfer_dist,
    dataloader,
    epoch,
    global_step,
    device,
    logger,
    writer,
    optimizer,
    vis_loader=None,
):
    """Training / Eval function."""
    modes = ["train", "eval"]
    if mode not in modes:
        raise ValueError(f"mode ({mode}) must be one of {modes}.")
    if mode == "train":
        deformer.train()
    else:
        deformer.eval()
    tot_loss = 0
    count = 0
    criterion = LOSSES[args.loss_type]
    epoch_images = []
    epoch_latents = []

    with torch.set_grad_enabled(mode == "train"):
        toc = time.time()

        for batch_idx, data_tensors in enumerate(dataloader):
            tic = time.time()
            # Send tensors to device.
            data_tensors = [t.to(device) for t in data_tensors]
            (
                ii,
                jj,
                source_pts,
                target_pts,
                source_img,
                target_img,
            ) = data_tensors

            bs = len(source_pts)
            optimizer.zero_grad()

            # Batch together source and target to create two-way loss training.
            # Cannot call deformer twice (once for each way) because that
            # breaks odeint_ajoint's gradient computation. not sure why.
            source_target_points = torch.cat([source_pts, target_pts], dim=0)
            target_source_points = torch.cat([target_pts, source_pts], dim=0)

            source_target_latents = torch.cat([ii, jj], dim=0)
            target_source_latents = torch.cat([jj, ii], dim=0)
            latent_seq = torch.stack(
                [source_target_latents, target_source_latents], dim=1
            )
            deformed_pts = deformer(
                source_target_points[..., :3], latent_seq
            )  # Already set to via_hub.

            if mode == "eval":
                # Add thumbnail images for visualizing latent embedding.
                epoch_images += [torch.cat([source_img, target_img], dim=0)]
                source_target_latents = deformer.module.get_lat_params(
                    source_target_latents
                )
                epoch_latents += [source_target_latents]

            # Symmetric pair of matching losses.
            if args.symm:
                _, _, dist = chamfer_dist(
                    utils.symmetric_duplication(deformed_pts, symm_dim=2),
                    utils.symmetric_duplication(
                        target_source_points[..., :3], symm_dim=2
                    ),
                )
            else:
                _, _, dist = chamfer_dist(
                    deformed_pts, target_source_points[..., :3]
                )

            loss = criterion(dist, torch.zeros_like(dist))

            # Check amount of deformation.
            deform_abs = torch.mean(
                torch.norm(deformed_pts - source_target_points, dim=-1)
            )

            if mode == "train":
                loss.backward()

                # Gradient clipping.
                torch.nn.utils.clip_grad_value_(
                    deformer.module.parameters(), args.clip_grad
                )

                optimizer.step()

            tot_loss += loss.item()
            count += bs

            if batch_idx % args.log_interval == 0:
                # Logger log.
                logger.info(
                    "{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t"
                    "Dist Mean: {:.6f}\t"
                    "Deform Mean: {:.6f}\t"
                    "DataTime: {:.4f}\tComputeTime: {:.4f}".format(
                        mode,
                        epoch,
                        batch_idx * bs,
                        len(dataloader) * bs,
                        100.0 * batch_idx / len(dataloader),
                        loss.item(),
                        np.sqrt(loss.item()),
                        deform_abs.item(),
                        tic - toc,
                        time.time() - tic,
                    )
                )
                # Tensorboard log.
                writer.add_scalar(
                    f"{mode}/loss_sum",
                    loss.item(),
                    global_step=int(global_step),
                )
                writer.add_scalar(
                    f"{mode}/dist_avg",
                    np.sqrt(loss.item()),
                    global_step=int(global_step),
                )
                writer.add_scalar(
                    f"{mode}/def_mean",
                    deform_abs.item(),
                    global_step=int(global_step),
                )

            if mode == "train":
                global_step += 1
            toc = time.time()
    tot_loss /= count

    # # visualize embeddings
    if mode == "eval":
        epoch_images = torch.cat(epoch_images, dim=0).permute(0, 3, 1, 2)  # \
        # [N,C,H,W]
        epoch_images = epoch_images.float() / 255.0
        epoch_latents = torch.cat(epoch_latents, dim=0)
        writer.add_embedding(
            mat=epoch_latents, label_img=epoch_images, global_step=epoch
        )

    # # visualize a few deformation examples in tensorboard
    if args.vis_mesh and (vis_loader is not None) and (mode == "eval"):
        # add deformation demo
        with torch.set_grad_enabled(False):
            for ind, data_tensors in enumerate(vis_loader):  # batch size = 1
                ii = torch.tensor([data_tensors[0]], dtype=torch.long)
                jj = torch.tensor([data_tensors[1]], dtype=torch.long)

                source_latents = deformer.module.get_lat_params(ii)
                target_latents = deformer.module.get_lat_params(jj)
                hub_latents = torch.zeros_like(source_latents)

                data_tensors = [
                    t.unsqueeze(0).to(device) for t in data_tensors[2:]
                ]
                vi, fi, vj, fj = data_tensors
                vi = vi[0]
                fi = fi[0]
                vj = vj[0]
                fj = fj[0]
                vi_j = deformer(
                    vi[..., :3],
                    torch.stack(
                        [source_latents, hub_latents, target_latents],
                        dim=1,
                    ),
                )
                vj_i = deformer(
                    vj[..., :3],
                    torch.stack(
                        [target_latents, hub_latents, source_latents],
                        dim=1,
                    ),
                )

                accu_i, _, _ = chamfer_dist(vi_j, vj)  # [1, m]
                accu_j, _, _ = chamfer_dist(vj_i, vi)  # [1, n]

                # Find the max dist between pairs of original shapes for
                # normalizing colors
                chamfer_dist.set_reduction_method("max")
                _, _, max_dist = chamfer_dist(vi, vj)  # [1,]
                chamfer_dist.set_reduction_method("mean")

                # Normalize the accuracies wrt. the distance between src
                # and tgt meshes.
                ci = utils.colorize_scalar_tensors(
                    accu_i / max_dist, vmin=0.0, vmax=1.0, cmap="coolwarm"
                )
                cj = utils.colorize_scalar_tensors(
                    accu_j / max_dist, vmin=0.0, vmax=1.0, cmap="coolwarm"
                )
                ci = (ci * 255.0).int()
                cj = (cj * 255.0).int()

                # Save mesh.
                samp_dir = os.path.join(args.log_dir, "deformation_samples")
                os.makedirs(samp_dir, exist_ok=True)
                trimesh.Trimesh(
                    vi.detach().cpu().numpy()[0], fi.detach().cpu().numpy()[0]
                ).export(os.path.join(samp_dir, f"samp{ind}_src.obj"))
                trimesh.Trimesh(
                    vj.detach().cpu().numpy()[0], fj.detach().cpu().numpy()[0]
                ).export(os.path.join(samp_dir, f"samp{ind}_tar.obj"))
                trimesh.Trimesh(
                    vi_j.detach().cpu().numpy()[0],
                    fi.detach().cpu().numpy()[0],
                ).export(os.path.join(samp_dir, f"samp{ind}_src_to_tar.obj"))
                trimesh.Trimesh(
                    vj_i.detach().cpu().numpy()[0],
                    fj.detach().cpu().numpy()[0],
                ).export(os.path.join(samp_dir, f"samp{ind}_tar_to_src.obj"))

                # Add colorized mesh to tensorboard.
                writer.add_mesh(
                    f"samp{ind}/src",
                    vertices=vi,
                    faces=fi,
                    global_step=int(epoch),
                )
                writer.add_mesh(
                    f"samp{ind}/tar",
                    vertices=vj,
                    faces=fj,
                    global_step=int(epoch),
                )
                writer.add_mesh(
                    f"samp{ind}/src_to_tar",
                    vertices=vi_j,
                    faces=fi,
                    colors=ci,
                    global_step=int(epoch),
                )
                writer.add_mesh(
                    f"samp{ind}/tar_to_src",
                    vertices=vj_i,
                    faces=fj,
                    colors=cj,
                    global_step=int(epoch),
                )

    return tot_loss


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ShapeNet Deformation Space")

    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--pseudo_train_epoch_size",
        type=int,
        default=2048,
        metavar="N",
        help="number of samples in an pseudo-epoch. (default: 2048)",
    )
    parser.add_argument(
        "--pseudo_eval_epoch_size",
        type=int,
        default=128,
        metavar="N",
        help="number of samples in an pseudo-epoch. (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="R",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/shapenet_simplified",
        help="path to mesh folder root (default: data/shapenet_simplified)",
    )
    parser.add_argument(
        "--category", type=str, default="chair", help="the shape category."
    )
    parser.add_argument(
        "--thumbnails_root",
        type=str,
        default="data/shapenet_thumbnails",
        help="path to thumbnails folder root "
        "(default: data/shapenet_thumbnails)",
    )
    parser.add_argument(
        "--deformer_arch",
        type=str,
        choices=["imnet", "vanilla"],
        default="imnet",
        help="deformer architecture. (default: imnet)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=SOLVERS,
        default="dopri5",
        help="ode solver. (default: dopri5)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="absolute error tolerence in ode solver. (default: 1e-5)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="relative error tolerence in ode solver. (default: 1e-5)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--log_dir", type=str, required=True, help="log directory for run"
    )
    parser.add_argument(
        "--nonlin", type=str, default="elu", help="type of nonlinearity to use"
    )
    parser.add_argument(
        "--optim", type=str, default="adam", choices=list(OPTIMIZERS.keys())
    )
    parser.add_argument(
        "--loss_type", type=str, default="l2", choices=list(LOSSES.keys())
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to checkpoint if resume is needed",
    )
    parser.add_argument(
        "-n",
        "--nsamples",
        default=2048,
        type=int,
        help="number of sample points to draw per shape.",
    )
    parser.add_argument(
        "--lat_dims", default=32, type=int, help="number of latent dimensions."
    )
    parser.add_argument(
        "--datasubset",
        default=0,
        type=int,
        help="0 to not subset. else subset this many examples.",
    )
    parser.add_argument(
        "--deformer_nf",
        default=100,
        type=int,
        help="number of base number of feature layers in deformer (imnet).",
    )
    parser.add_argument(
        "--lr_scheduler", dest="lr_scheduler", action="store_true"
    )
    parser.add_argument(
        "--no_lr_scheduler", dest="lr_scheduler", action="store_false"
    )
    parser.set_defaults(lr_scheduler=True)
    parser.set_defaults(normals=True)
    parser.add_argument(
        "--visualize_mesh",
        dest="vis_mesh",
        action="store_true",
        help="visualize deformation for meshes of sample validation data "
        "in tensorboard.",
    )
    parser.add_argument(
        "--no_visualize_mesh",
        dest="vis_mesh",
        action="store_false",
        help="no visualize deformation for meshes of sample validation data "
        "in tensorboard.",
    )
    parser.set_defaults(vis_mesh=True)
    parser.add_argument(
        "--adjoint",
        dest="adjoint",
        action="store_true",
        help="use adjoint solver to propagate gradients thru odeint.",
    )
    parser.add_argument(
        "--no_adjoint",
        dest="adjoint",
        action="store_false",
        help="not use adjoint solver to propagate gradients thru odeint.",
    )
    parser.set_defaults(adjoint=True)
    parser.add_argument(
        "--sign_net",
        dest="sign_net",
        action="store_true",
        help="use sign net.",
    )
    parser.add_argument(
        "--no_sign_net",
        dest="sign_net",
        action="store_false",
        help="not use sign net.",
    )
    parser.set_defaults(sign_net=False)
    parser.add_argument(
        "--clip_grad",
        default=1.0,
        type=float,
        help="clip gradient to this value. large value basically "
        "deactivates it.",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        choices=[
            "nn_replace",
            "nn_no_replace",
            "all_replace",
            "all_no_replace",
        ],
        default="nn_no_replace",
        help="method for sampling pairs of shape to deform.",
    )
    parser.add_argument(
        "--symm", dest="symm", action="store_true", help="use symmetric flow."
    )
    parser.add_argument(
        "--no_symm",
        dest="symm",
        action="store_false",
        help="not use symmetric flow.",
    )
    parser.set_defaults(symm=False)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Adjust batch size based on the number of gpus available.
    args.batch_size = int(torch.cuda.device_count()) * args.batch_size_per_gpu
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    kwargs = (
        {"num_workers": min(12, args.batch_size), "pin_memory": True}
        if use_cuda
        else {}
    )
    device = torch.device("cuda" if use_cuda else "cpu")

    # Log and create snapshots.
    filenames_to_snapshot = (
        glob.glob("*.py") + glob.glob("*.sh") + glob.glob("layers/*.py")
    )
    utils.snapshot_files(filenames_to_snapshot, args.log_dir)
    logger = utils.get_logger(log_dir=args.log_dir)
    with open(os.path.join(args.log_dir, "params.json"), "w") as fh:
        json.dump(args.__dict__, fh, indent=2)
    logger.info("%s", repr(args))

    args.n_vis = 2  # Number of deformation examples to visualize.

    # Tensorboard writer.
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tensorboard"))

    # Random seed for reproducability.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create dataloaders.
    fullset = dl.ShapeNetVertex(
        data_root=args.data_root,
        split="train",
        category=args.category,
        nsamples=args.nsamples,
        normals=False,
    )
    if args.datasubset > 0:
        fullset.restrict_subset(args.datasubset)

    # Return thumbnails (to visualize embedding during eval).
    fullset.add_thumbnails(args.thumbnails_root)

    if "nn_" in args.sampling_method:
        args.nn_samp = True
        replace = True if args.sampling_method == "nn_replace" else False
        train_sampler = dl.LatentNearestNeighborSampler(
            dataset=fullset,
            src_split="train",
            tar_split="train",
            n_samples=args.pseudo_train_epoch_size,
            k=1000,
            replace=replace,
        )
        eval_sampler = dl.LatentNearestNeighborSampler(
            dataset=fullset,
            src_split="train",
            tar_split="train",
            n_samples=args.pseudo_eval_epoch_size,
            k=1,
            replace=replace,
        )  # Pick the closest.
        vis_sampler = dl.LatentNearestNeighborSampler(
            dataset=fullset,
            src_split="train",
            tar_split="train",
            n_samples=args.n_vis,
            k=1,
            replace=replace,
        )  # Pick the closest.
    else:
        args.nn_samp = False
        replace = True if args.sampling_method == "all_replace" else False
        train_sampler = dl.RandomPairSampler(
            dataset=fullset,
            src_split="train",
            tar_split="train",
            n_samples=args.pseudo_train_epoch_size,
            replace=replace,
        )
        eval_sampler = dl.RandomPairSampler(
            dataset=fullset,
            src_split="train",
            tar_split="train",
            n_samples=args.pseudo_eval_epoch_size,
            replace=replace,
        )
        vis_sampler = dl.RandomPairSampler(
            dataset=fullset,
            src_split="train",
            tar_split="train",
            n_samples=args.n_vis,
            replace=replace,
        )

    # Make sure we are turning off shuffle since we are using samplers!
    train_loader = DataLoader(
        fullset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        sampler=train_sampler,
        **kwargs,
    )
    eval_loader = DataLoader(
        fullset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        sampler=eval_sampler,
        **kwargs,
    )

    if args.vis_mesh:
        # For loading full meshes for visualization.
        simp_data_root = args.data_root
        simpset = dl.ShapeNetMesh(
            data_root=simp_data_root,
            split="train",
            category=args.category,
            normals=False,
        )
        if args.datasubset > 0:
            simpset.restrict_subset(args.datasubset)
        if not (
            vis_sampler.dataset.fname_to_idx_dict == simpset.fname_to_idx_dict
        ):
            raise RuntimeError(
                f"vis_sampler ({len(vis_sampler.dataset.fname_to_idx_dict)}) "
                f"does not match sample set ({len(simpset.fname_to_idx_dict)})"
            )
        vis_loader = DataLoader(
            simpset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            sampler=vis_sampler,
            **kwargs,
        )
    else:
        vis_loader = None

    # Setup model.
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

    # Awkward workaround to get gradients from odeint_adjoint to lat_params.
    lat_params = torch.nn.Parameter(
        torch.randn(fullset.n_shapes, args.lat_dims) * 1e-1, requires_grad=True
    )
    deformer.add_lat_params(lat_params)
    deformer.to(device)

    all_model_params = list(deformer.parameters())

    optimizer = OPTIMIZERS[args.optim](all_model_params, lr=args.lr)

    start_ep = 0
    global_step = np.zeros(1, dtype=np.uint32)
    tracked_stats = np.inf

    if args.resume:
        logger.info(
            "Loading checkpoint {} ================>".format(args.resume)
        )
        resume_dict = torch.load(args.resume)
        start_ep = resume_dict["epoch"]
        global_step = resume_dict["global_step"]
        tracked_stats = resume_dict["tracked_stats"]
        deformer.load_state_dict(resume_dict["deformer_state_dict"])
        optimizer.load_state_dict(resume_dict["optim_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        logger.info("[!] Successfully loaded checkpoint.")

    # More threads don't seem to help.
    chamfer_dist = ChamferDistKDTree(reduction="mean", njobs=1)
    chamfer_dist.to(device)
    deformer = nn.DataParallel(deformer)
    deformer.to(device)

    model_param_count = lambda model: sum(  # noqa: E731
        x.numel() for x in model.parameters()
    )
    logger.info(
        f"{model_param_count(deformer)}(deformer) paramerters in total"
    )

    checkpoint_path = os.path.join(args.log_dir, "checkpoint_latest.pth.tar")

    if args.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    train_latent_dict = compute_latent_dict(deformer, fullset)
    # Training loop.
    for epoch in range(start_ep + 1, args.epochs + 1):
        # Set sampler nn graph before train or eval.
        if args.nn_samp:
            train_loader.sampler.update_nn_graph(
                train_latent_dict, train_latent_dict, k=get_k(epoch)
            )
            eval_loader.sampler.update_nn_graph(
                train_latent_dict, train_latent_dict
            )
            vis_loader.sampler.update_nn_graph(
                train_latent_dict, train_latent_dict
            )

        _ = train_or_eval(
            "train",
            args,
            deformer,
            chamfer_dist,
            train_loader,
            epoch,
            global_step,
            device,
            logger,
            writer,
            optimizer,
            None,
        )
        loss_eval = train_or_eval(
            "eval",
            args,
            deformer,
            chamfer_dist,
            eval_loader,
            epoch,
            global_step,
            device,
            logger,
            writer,
            optimizer,
            vis_loader,
        )

        if args.lr_scheduler:
            scheduler.step(loss_eval)
        if loss_eval < tracked_stats:
            tracked_stats = loss_eval
            is_best = True
        else:
            is_best = False

        utils.save_checkpoint(
            {
                "epoch": epoch,
                "deformer_state_dict": deformer.module.state_dict(),
                "lat_params": lat_params,
                "optim_state_dict": optimizer.state_dict(),
                "tracked_stats": tracked_stats,
                "global_step": global_step,
            },
            is_best,
            epoch,
            checkpoint_path,
            "_shapeflow",
            logger,
        )


if __name__ == "__main__":
    main()
