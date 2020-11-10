# ShapeFlow
[NeurIPS 20'] ShapeFlow: Learnable Deformations Among 3D Shapes.

 
By: [Chiyu "Max" Jiang*](http://maxjiang.ml/), [Jingwei Huang*](http://stanford.edu/~jingweih/), [Andrea Tagliasacchi](http://gfx.uvic.ca/people/ataiya/), [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/)

\[[Project Website]()\] \[[Paper](https://arxiv.org/abs/2006.07982)\]

<img src="doc/teaser.png" alt="deepdeform_teaser" width="1000"/>


## 1. Introduction
We present ShapeFlow, a flow-based model for learning a deformation space for entire classes of 3D shapes with large intra-class variations. ShapeFlow allows learning a multi-template deformation space that is agnostic to shape topology, yet preserves fine geometric details. Different from a generative space where a latent vector is directly decoded into a shape, a deformation space decodes a vector into a continuous flow that can advect a source shape towards a target. Such a space naturally allows the disentanglement of geometric style (coming from the source) and structural pose (conforming to the target). We parametrize the deformation between geometries as a learned continuous flow field via a neural network and show that such deformations can be guaranteed to have desirable properties, such as bijectivity, freedom from self-intersections, or volume preservation. We illustrate the effectiveness of this learned deformation space for various downstream applications, including shape generation via deformation, geometric style transfer, unsupervised learning of a consistent parameterization for entire classes of shapes, and shape interpolation.

## 2. Getting Started
### Installing dependencies
We recommend using pip to install all required dependencies with ease.
```
pip install -r requirements.txt
```

### Optional dependency (rendering)
We strongly suggest installing the optional dependencies for rendering meshes, so that you can visualize the results using interactive notebooks.
`pyrender` can be installed via pip:
```
pip install pyrender
```

Additionally to run the notebook renderings on a headless server, follow the [instructions](https://pyrender.readthedocs.io/en/latest/install/#python-installation) for installing `OSMesa`.

### Download and unpack data
To download and unpack the data used in the experiment, please use the utility script privided.
```
bash download_data.sh
```

## 3. Reproduce results
### Run experiments
Please use our provided launch script to start training the shape deformation model.
```
bash shapenet_train.sh
```

The training will launch on all available GPUs. Mask GPUs accordingly if you want to use only a subset of all GPUs. The initial tests are done on NVIDIA Volta V100 GPUs, therefore the `batch_size_per_gpu=16` might need to be adjusted accordingly for GPUs with smaller or larger memory limits if the out of memory error is triggered.

### Load and visualize pretrained checkpoint
First download the pretrained checkpoint.
```
wget island.me.berkeley.edu/files/pretrained_ckpt.zip
mkdir -p runs
mv pretrained_ckpt.zip runs
cd runs; unzip pretrained_ckpt.zip; rm pretrained_ckpt.zip; cd ..
```

Next, run through the cells in `visualize_deformer.ipynb`.

### Monitor training
After launching the training script, a `runs` directory will be created, with different runs each as a separate subfolder within. To monitor the training process based on text logs, use
```
tail -f runs/<job_name>/log.txt
```

To monitor the training process using tensorboard, do:
```
# if you are running this on a remote server via ssh
ssh my_favorite_machine -L 6006:localhost:6006

# go to the directory containing the tensorboard log
cd path/to/ShapeFlow/runs/<job_name>/tensorboard

# launch tensorboard
tensorboard --logdir . --port 6006
```
Tensorboard allows tracking of deformation losses, as well as visualizing the source / target / deformed meshes. The deformed meshes are colored by the distance per vertex with respect to target shape.

<img src="doc/tb_losses.png" alt="tensorboard_losses" width="500"/>
<img src="doc/tb_meshes.png" alt="tensorboard_meshes" width="500"/>

### Citation
If you find our code useful for your work, please consider citing our paper:
```
@inproceedings{jiang2020shapeflow,
  title={ShapeFlow: Learnable Deformations Among 3D Shapes},
  author={Jiang, Chiyu and Huang, Jingwei and Tagliasacchi, Andrea and Guibas, Leonidas},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

### Contact
Please contact [Max Jiang](mailto:maxjiang93@gmail.com) if you have further questions!
