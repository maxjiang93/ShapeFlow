"""ShapeNet deformation dataloader"""
import os
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import trimesh
import glob
import imageio
import pickle
from collections import OrderedDict
from scipy.spatial import cKDTree


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

SPLITS = ["train", "test", "val", "*"]


def strip_name(filename):
    if len(filename.split("/")) > 3:
        return "/".join(filename.split("/")[-4:-1])
    else:
        return filename


class ShapeNetBase(Dataset):
    """Pytorch Dataset base for loading ShapeNet shape pairs."""

    def __init__(self, data_root, split, category="chair"):
        """
        Initialize DataSet
        Args:
          data_root: str, path to data root that contains the ShapeNet dataset.
          split: str, one of 'train'/'val'/'test'/'*'. '*' for all splits.
          catetory:
            str, name of the category to train on. 'all' for all 13 classes.
            Otherwise can be a comma separated string containing multiple names
        """
        self.data_root = data_root
        self.split = split

        if not (split in SPLITS):
            raise ValueError(f"{split} must be one of {SPLITS}")
        self.categories = [c.strip() for c in category.split(",")]
        cats = list(cat_to_synset.keys())
        if "all" in self.categories:
            self.categories = cats
        for c in self.categories:
            if c not in cats:
                raise ValueError(
                    f"{c} is not in the list of the 13 categories: {cats}"
                )
        self.files = self._get_filenames(
            self.data_root, self.split, self.categories
        )
        self._file_splits = None

        self.thumbnails_dir = None
        self.thumbnails = False
        self._fname_to_idx_dict = None

    @property
    def file_splits(self):
        if self._file_splits is None:
            self._file_splits = {"train": [], "test": [], "val": []}
            for f in self.files:
                if "train/" in f:
                    self._file_splits["train"].append(f)
                elif "test/" in f:
                    self._file_splits["test"].append(f)
                else:  # val/
                    self._file_splits["val"].append(f)

        return self._file_splits

    @staticmethod
    def _get_filenames(data_root, split, categories):
        files = []
        for c in categories:
            synset_id = cat_to_synset[c]
            if split != "*":
                cat_folder = os.path.join(data_root, split, synset_id)
                if not os.path.exists(cat_folder):
                    raise RuntimeError(
                        f"Datafolder for {synset_id} ({c}) "
                        f"does not exist at {cat_folder}."
                    )
                files += glob.glob(
                    os.path.join(cat_folder, "*/*.ply"), recursive=True
                )
            else:
                for split in SPLITS[:3]:
                    cat_folder = os.path.join(data_root, split, synset_id)
                    if not os.path.exists(cat_folder):
                        raise RuntimeError(
                            f"Datafolder for {synset_id} ({c}) does not exist "
                            f"at {cat_folder}."
                        )
                    files += glob.glob(
                        os.path.join(cat_folder, "*/*.ply"), recursive=True
                    )
        return sorted(files)

    def __len__(self):
        return self.n_shapes ** 2

    @property
    def n_shapes(self):
        return len(self.files)

    def restrict_subset(self, indices):
        """Restrict data to the subset of data as indicated by the indices.

        Mostly helpful for debugging only.

        Args:
          indices: list or array of ints, to index the original self.files
        """
        self.files = [self.files[i] for i in indices]

    @property
    def fname_to_idx_dict(self):
        """A dict mapping unique mesh names to indicies."""
        if self._fname_to_idx_dict is None:
            fnames = ["/".join(f.split("/")[-4:-1]) for f in self.files]
            self._fname_to_idx_dict = dict(
                zip(fnames, list(range(len(fnames))))
            )
        return self._fname_to_idx_dict

    def idx_to_combinations(self, idx):
        """Convert s linear index to a pair of indices."""
        i = np.floor(idx / self.n_shapes)
        j = idx - i * self.n_shapes
        if hasattr(idx, "__len__"):
            i = np.array(i, dtype=int)
            j = np.array(j, dtype=int)
        else:
            i = int(i)
            j = int(j)
        return i, j

    def combinations_to_idx(self, i, j):
        """Convert a pair of indices to a linear index."""
        idx = i * self.n_shapes + j
        if hasattr(idx, "__len__"):
            idx = np.array(idx, dtype=int)
        else:
            idx = int(idx)
        return idx


class ShapeNetVertex(ShapeNetBase):
    """Pytorch Dataset for sampling vertices from meshes."""

    def __init__(
        self, data_root, split, category="chair", nsamples=5000, normals=True
    ):
        """
        Initialize DataSet
        Args:
          data_root: str, path to data root that contains the ShapeNet dataset.
          split: str, one of 'train'/'val'/'test'.
          catetory: str, name of the category to train on. 'all' for all 13
            classes. Otherwise can be a comma separated string containing
            multiple names.
          nsamples: int, number of points to sample from each mesh.
          normals: bool, whether to add normals to the point features.
        """
        super(ShapeNetVertex, self).__init__(
            data_root=data_root, split=split, category=category
        )
        self.nsamples = nsamples
        self.normals = normals

    @staticmethod
    def sample_mesh(mesh_path, nsamples, normals=True):
        """Load the mesh from mesh_path and sample nsampels points from its vertices.

        If nsamples < number of vertices on mesh, randomly repeat some
        vertices as padding.

        Args:
          mesh_path: str, path to load the mesh from.
          nsamples: int, number of vertices to sample.
          normals: bool, whether to add normals to the point features.
        Returns:
          v_sample: np array of shape [nsamples, 3 or 6] for sampled points.
        """
        mesh = trimesh.load(mesh_path)
        v = np.array(mesh.vertices)
        nv = v.shape[0]
        seq = np.random.permutation(nv)[:nsamples]
        if len(seq) < nsamples:
            seq_repeat = np.random.choice(
                nv, nsamples - len(seq), replace=True
            )
            seq = np.concatenate([seq, seq_repeat], axis=0)
        v_sample = v[seq]
        if normals:
            n_sample = np.array(mesh.vertex_normals[seq])
            v_sample = np.concatenate([v_sample, n_sample], axis=-1)

        return v_sample

    def add_thumbnails(self, thumbnails_root):
        self.thumbnails = True
        self.thumbnails_dir = thumbnails_root

    def _get_one_mesh(self, idx):
        verts = self.sample_mesh(self.files[idx], self.nsamples, self.normals)
        verts = verts.astype(np.float32)
        if self.thumbnails:
            thumb_dir = self.files[idx].replace(
                self.data_root, self.thumbnails_dir
            )
            thumb_dir = os.path.dirname(thumb_dir)
            thumb_file = os.path.join(thumb_dir, "thumbnail.jpg")
            thumb = np.array(imageio.imread(thumb_file))
            return verts, thumb
        else:
            return verts

    def __getitem__(self, idx):
        """Get a random pair of shapes corresponding to idx.
        Args:
          idx: int, index of the shape pair to return. must be smaller than
            len(self).
        Returns:
          verts_i: [npoints, 3 or 6] float tensor for point samples from the
            first mesh.
          verts_j: [npoints, 3 or 6] float tensor for point samples from the
            second mesh.
          thumb_i: (optional) [H, W, 3] int8 tensor for thumbnail image for
            the first mesh.
          thumb_j: (optional) [H, W, 3] int8 tensor for thumbnail image for
            the second mesh.
        """
        i, j = self.idx_to_combinations(idx)
        if self.thumbnails:
            verts_i, thumb_i = self._get_one_mesh(i)
            verts_j, thumb_j = self._get_one_mesh(j)
            return i, j, verts_i, verts_j, thumb_i, thumb_j
        else:
            verts_i = self._get_one_mesh(i)
            verts_j = self._get_one_mesh(j)
            return i, j, verts_i, verts_j


class ShapeNetMesh(ShapeNetBase):
    """Pytorch Dataset for sampling entire meshes."""

    def __init__(self, data_root, split, category="chair", normals=True):
        """
        Initialize DataSet
        Args:
          data_root: str, path to data root that contains the ShapeNet dataset.
          split: str, one of 'train'/'val'/'test'.
          catetory:
            str, name of the category to train on. 'all' for all 13 classes.
            Otherwise can be a comma separated string containing multiple
            names.
        """
        super(ShapeNetMesh, self).__init__(
            data_root=data_root, split=split, category=category
        )
        self.normals = normals

    def get_pairs(self, i, j):
        verts_i, faces_i = self.get_single(i)
        verts_j, faces_j = self.get_single(j)

        return i, j, verts_i, faces_i, verts_j, faces_j

    def get_single(self, i):
        mesh_i = trimesh.load(self.files[i])

        verts_i = mesh_i.vertices.astype(np.float32)
        faces_i = mesh_i.faces.astype(np.int32)

        if self.normals:
            norms_i = mesh_i.vertex_normals.astype(np.float32)
            verts_i = np.concatenate([verts_i, norms_i], axis=-1)

        verts_i = torch.from_numpy(verts_i)
        faces_i = torch.from_numpy(faces_i)

        return verts_i, faces_i

    def __getitem__(self, idx):
        """Get a random pair of meshes.
        Args:
          idx: int, index of the shape pair to return. must be smaller than
            len(self).
        Returns:
          verts_i: [#vi, 3 or 6] float tensor for vertices from the first mesh.
          faces_i: [#fi, 3 or 6] int32 tensor for faces from the first mesh.
          verts_j: [#vj, 3 or 6] float tensor for vertices from the 2nd mesh.
          faces_j: [#fj, 3 or 6] int32 tensor for faces from the 2nd mesh.
        """
        i, j = self.idx_to_combinations(idx)
        return self.get_pairs(i, j)


class FixedPointsCachedDataset(Dataset):
    """Dataset for loading fixed points dataset from cached pickle file."""

    def __init__(self, pkl_file, npts=1024):
        with open(pkl_file, "rb") as fh:
            self.data_dict = pickle.load(fh)
        self.data_dict = OrderedDict(sorted(self.data_dict.items()))
        self.key_list = list(self.data_dict.keys())
        assert npts <= 4096 and npts > 0
        self.npts = npts

    def __getitem__(self, idx):
        filename = self.key_list[idx]
        points = self.data_dict[filename]
        rand_seq = np.random.choice(points.shape[0], self.npts, replace=False)
        points_ = points[rand_seq]
        return filename, idx, points_

    def __len__(self):
        return len(self.data_dict)


class PairSamplerBase(Sampler):
    """Data sampler base for sampling pairs."""

    def __init__(
        self, dataset, src_split, tar_split, n_samples, replace=False
    ):
        assert src_split in SPLITS[:3]
        assert tar_split in SPLITS[:3]
        self.replace = replace
        self.n_samples = n_samples
        self.src_split = src_split
        self.tar_split = tar_split
        self.dataset = dataset
        self.src_files = self.dataset.file_splits[src_split]
        self.tar_files = self.dataset.file_splits[tar_split]
        self.src_files = [strip_name(f) for f in self.src_files]
        self.tar_files = [strip_name(f) for f in self.tar_files]
        self.n_src = len(self.src_files)
        self.n_tar = len(self.tar_files)
        if not replace:
            if not self.n_samples <= self.n_src:
                raise RuntimeError(
                    f"Numer of samples ({len(self.n_samples)}) must be "
                    f"less than number source shapes ({len(self.n_src)})"
                )

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class RandomPairSampler(PairSamplerBase):
    """Data sampler for sampling random pairs."""

    def __init__(
        self, dataset, src_split, tar_split, n_samples, replace=False
    ):
        super(RandomPairSampler, self).__init__(
            dataset, src_split, tar_split, n_samples, replace
        )

    def __iter__(self):
        d = self.dataset
        if self.replace:
            src_names = np.random.choice(
                self.src_files, self.n_samples, replace=True
            )
            tar_names = np.random.choice(
                self.tar_files, self.n_samples, replace=True
            )
        else:
            src_names = np.random.permutation(self.src_files)[
                : int(self.n_samples)
            ]
            tar_names = np.random.permutation(self.tar_files)[
                : int(self.n_samples)
            ]
        src_idxs = np.array(
            [d.fname_to_idx_dict[strip_name(f)] for f in src_names]
        )
        tar_idxs = np.array(
            [d.fname_to_idx_dict[strip_name(f)] for f in tar_names]
        )
        combo_ids = self.dataset.combinations_to_idx(src_idxs, tar_idxs)

        return iter(combo_ids)

    def __len__(self):
        return self.n_samples


class LatentNearestNeighborSampler(PairSamplerBase):
    """Data sampler for sampling pairs from top-k nearest latent neighbors."""

    def __init__(
        self, dataset, src_split, tar_split, n_samples, k, replace=False
    ):
        """Initialize.

        Args:
          k: int, top-k neighbors to sample from.
          replace: bool, sample with replacement.
                   if no replace, then must ensure n_samples <= n_shapes
        """
        super(LatentNearestNeighborSampler, self).__init__(
            dataset, src_split, tar_split, n_samples, replace
        )
        self.k = k
        self.graph_set = False

    def update_nn_graph(self, src_latent_dict, tar_latent_dict, k=None):
        """Update nearest neighbor graph.

        Args:
          src_latent_dict: a dict that maps filenames to latent codes for
            source set.
          tar_latent_dict: a dict that maps filenames to latent codes for
            target set.
        """
        if k is not None:
            self.k = k
        tar_names = list(tar_latent_dict.keys())
        tar_latents = list(tar_latent_dict.values())
        tar_latents = np.stack(tar_latents, axis=0)  # [n, lat_dim]
        # build kd-tree to accelerate nearest neighbor computation
        k = self.k + 1 if self.src_split == self.tar_split else self.k
        self._kdtree = cKDTree(tar_latents)

        src_names = list(src_latent_dict.keys())
        src_latents = list(src_latent_dict.values())
        src_latents = np.stack(src_latents, axis=0)  # [m, lat_dim]
        _, nn_idx = self._kdtree.query(src_latents, k=k)  # [m, k]
        if nn_idx.ndim == 1:
            nn_idx = nn_idx[:, None]
        nn_idx = nn_idx[:, -self.k:]

        nn_names = []
        for i in range(nn_idx.shape[0]):
            nn_names.append([tar_names[j] for j in nn_idx[i]])
        self._nn_map = dict(zip(src_names, nn_names))
        self.graph_set = True

    @property
    def kdtree(self):
        return self._kdtree

    @property
    def nn_map(self):
        return self._nn_map

    def __iter__(self):
        if not self.graph_set:
            raise RuntimeError(
                "Nearest neighbor graph not yet set."
                " Run '.update_nn_graph()' to update first."
            )
        d = self.dataset
        # return generator
        if self.replace:
            src_names = np.random.choice(
                self.src_files, self.n_samples, replace=True
            )
        else:
            src_names = np.random.permutation(self.src_files)[
                : int(self.n_samples)
            ]

        for src_name in src_names:
            tar_name = np.random.choice(self.nn_map[src_name], 1)[0]
            src_idx = d.fname_to_idx_dict[strip_name(src_name)]
            tar_idx = d.fname_to_idx_dict[strip_name(tar_name)]
            combo_id = d.combinations_to_idx(src_idx, tar_idx)
            yield combo_id

    def __len__(self):
        return self.n_samples
