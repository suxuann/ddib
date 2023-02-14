# Dual Diffusion Implicit Bridges (ICLR 2023)

[**Dual Diffusion Implicit Bridges for Image-to-Image Translation**](https://openreview.net/forum?id=5HLoTvVGDe)<br/>
[Xuan Su](https://github.com/suxuann/ddib),
[Jiaming Song](https://tsong.me/),
[Chenlin Meng](https://cs.stanford.edu/~chenlin/),
[Stefano Ermon](https://cs.stanford.edu/~ermon/)<br/>
_[ICLR '23](https://openreview.net/forum?id=5HLoTvVGDe) |
[GitHub](https://github.com/suxuann/ddib) | [arXiv](https://arxiv.org/abs/2203.08382)
| [Colab](https://colab.research.google.com/drive/1-AC-z3DKSpgtCwbt7gASSGNtQOFM0BT6?usp=sharing)
| [Project Page](https://suxuann.github.io/ddib/)_

<img src="assets/figure_1.png" height="240" />

## Overview

Common image-to-image translation methods rely on joint training over data from both source and target domains. The
training process requires concurrent access to both datasets, which hinders data separation and privacy protection; and
existing models cannot be easily adapted to translation of new domain pairs. We present Dual Diffusion Implicit
Bridges (DDIBs), an image translation method based on diffusion models, that circumvents training on domain pairs. Image
translation with DDIBs relies on two diffusion models trained independently on each domain, and is a two-step process:
DDIBs first obtain latent encodings for source images with the source diffusion model, and then decode such encodings
using the target model to construct target images. Both steps are defined via ordinary differential equations (ODEs),
thus the process is cycle consistent only up to discretization errors of the ODE solvers. Theoretically, we interpret
DDIBs as concatenation of source to latent, and latent to target Schrödinger Bridges, a form of entropy-regularized
optimal transport, to explain the efficacy of the method. Experimentally, we apply DDIBs on synthetic and
high-resolution image datasets, to demonstrate their utility in a wide variety of translation tasks and their inherent
optimal transport properties.

## Installation

Installation follows the same procedures as in the above repositories.

We first install the current repository, and then install other libraries like `numpy, matplotlib` etc. My successful
installation contains the following version numbers, with Python 3.9:

```commandline
pip install -e .
pip install numpy==1.24.0 matplotlib==3.6.2 scikit-image==0.19.3 scikit-learn==1.2.0 gdown==4.6.0
conda install -c conda-forge mpi4py openmpi
```

## Synthetic Models

We release pretrained checkpoints for the 2D synthetic models in the paper.

### Installation

**Downloading via script**: In your repository, run `python download.py --exp synthetic` to download the pretrained
synthetic models. The
script will create a directory `models/synthetic` and automatically download the checkpoints to the directory.

**Downloading manually**: As an alternative, you can also download the checkpoint manually. Here is the download link
for the model
checkpoints: [Synthetic Models](https://drive.google.com/drive/folders/1YRP6nt96OJUOzEYY6N_Qh5xb3wEVFSjg?usp=sharing)

**Indexes**. We use indexes 0-5 to refer to the 6 synthetic types,
in: [Moons, Checkerboards, Concentric Rings, Concentric Squares, Parallel Rings, Parallel Squares].

**How are the datasets generated?** The key file to look at is: `guided_diffusion/synthetic_datasets.py`. We implement
the data generation and sampling processes for various 2D modalities.

After running the download script, we can run the cycle consistency, synthetic translation and sampling experiments
below.

### Training Synthetic Models

`python scripts/synthetic_train.py --num_res_blocks 3 --diffusion_steps 4000 --noise_schedule linear --lr 1e-4 --batch_size 20000 --task 0`

Task is an integer in {0, 1, 2, 3, 4, 5} corresponding to one of the synthetic types.

Training each model probably takes only a few (3-4) hours on a GPU.

The models and logs are saved to the directory at `OPENAI_LOGDIR`. If you want to save the model files to your desired
folder, modify the variable via `export OPENAI_LOGDIR=...`

### Cycle Consistency

`python scripts/synthetic_cycle.py --num_res_blocks 3 --diffusion_steps 4000 --batch_size 30000 --source 0 --target 1`

The above command runs the cycle-consistent translation experiment in the paper: between datasets Moons (0) and
Checkerboards (1). The generated experiment plots are saved under the new directory `experiments/images`.

### Synthetic Translation

`python scripts/synthetic_translation.py --num_res_blocks 3 --diffusion_steps 4000 --batch_size 30000 --source 0 --target 3`

The above command performs translation between the two synthetic domains and saves the resulting plots
to `experiments/images`.

### Sample from Synthetic Models

`python scripts/synthetic_sample.py --num_res_blocks 3 --diffusion_steps 4000 --batch_size 20000 --num_samples 80000 --task 1`

### Miscellaneous

- If you encounter `ModuleNotFoundError: No module named 'guided_diffusion'`, here is one possible
  solution: https://stackoverflow.com/a/23210066/3284573.

## ImageNet Translation

### Installation

**Download the model weights**. Similarly, run `python download.py --exp imagenet` to download the pretrained,
class-conditional ImageNet models from [guided-diffusion](https://github.com/openai/guided-diffusion). The script will
create a directory `models/imagenet` and put the classifier and diffusion model weights there.

**Copy the validation dataset**. We use the ImageNet validation set for domain translation. Two steps:

- Download the ImageNet validation set from ILSVRC2012. Unzip it. This will create a folder containing image files named
  like "ILSVRC2012_val_000XXXXX.JPG".
- Remember the path to the folder containing the validation set as `val_dir`, as we'll need it for the next command.

We are now ready to translate the images.

### Translation between ImageNet Classes

```commandline
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python scripts/imagenet_translation.py $MODEL_FLAGS --classifier_scale 1.0 --source 260,261,282,283 --target 261,262,283,284 --val_dir path_to_val_set
```

The above command copies the validation images for the source classes to `./experiments/imagenet`. Translated images are
placed in the same folder, with the target class appended in the filename.

We can update `source` and `target` to translate between other ImageNet classes. The corresponding val images are copied
automatically.

We translate the domain pairs in the specified order. For example, in the above command, we translate from class 260 to
261, 283 to 284, etc.

We can experiment with `classifier_scale` to guide the denoising process towards the target class with different
strengths.

We can prepend the Python command with `mpiexec -n N` to run it over multiple GPUs. For details, refer
to [guided-diffusion](https://github.com/openai/guided-diffusion).

## References and Acknowledgements

```
@inproceedings{
      su2022dual,
      title={Dual Diffusion Implicit Bridges for Image-to-Image Translation},
      author={Su, Xuan and Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
      booktitle={International Conference on Learning Representations},
      year={2023},
}
```

This implementation is based on / inspired by:
OpenAI: [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
and [openai/improved-diffusion](https://github.com/openai/improved-diffusion).

## To-do List

* Release pretrained models on AFHQ and yosemite datasets
* Add color translation experiments
* Add scripts to translate between AFHQ, yosemite images
