# Dual Diffusion Implicit Bridges

[**Dual Diffusion Implicit Bridges for Image-to-Image Translation**](https://openreview.net/forum?id=5HLoTvVGDe)<br/>
[Xuan Su](https://github.com/suxuann/ddib),
[Jiaming Song](https://tsong.me/),
[Chenlin Meng](https://cs.stanford.edu/~chenlin/),
[Stefano Ermon](https://cs.stanford.edu/~ermon/)<br/>
_[ICLR '23 Submission](https://openreview.net/forum?id=5HLoTvVGDe) |
[GitHub](https://github.com/suxuann/ddib) | [arXiv](https://arxiv.org/abs/2203.08382)
| [Project Page](https://github.com/suxuann/ddib#)_

<img src="assets/figure_1.png" height="240" />

This repository is heavily based on the repositories from
OpenAI: [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
and [openai/improved-diffusion](https://github.com/openai/improved-diffusion).

## Installation

Installation follows the same procedures as in the above repositories.

We first install the current repository, and then install other libraries like `numpy, matplotlib` etc. My successful
installation contains the following version numbers, with Python 3.9:

```commandline
pip install -e .
pip install numpy==1.24.0 matplotlib==3.6.2 scikit-image==0.19.3 scikit-learn==1.2.0 
conda install -c conda-forge mpi4py openmpi
```

## Synthetic Models

We release pretrained checkpoints for the 2D synthetic models in the paper.

Here is the download link for the model
checkpoints: [Synthetic Models](https://drive.google.com/drive/folders/1YRP6nt96OJUOzEYY6N_Qh5xb3wEVFSjg?usp=sharing)

**Indexes**. We use indexes 0-5 to refer to the 6 synthetic types,
in: [Moons, Checkerboards, Concentric Rings, Concentric Squares, Parallel Rings, Parallel Squares].

**How are the datasets generated?** The key file to look at is: `guided_diffusion/synthetic_datasets.py`. We implement
the data generation and sampling processes for various 2D modalities.

### Installation

In your repository, run `python download.py --experiment synthetic` to download the pretrained synthetic models. The
script will create a directory `models/synthetic` and download the checkpoints there.

After running the download script, we can run the cycle consistency, synthetic translation and sampling experiments
below.

### Training Synthetic Models

`python scripts/synthetic_train.py --num_res_blocks 3 --diffusion_steps 4000 --noise_schedule linear --lr 1e-4 --batch_size 20000 --task 0`

Task is an integer in {0, 1, 2, 3, 4, 5} corresponding to one of the synthetic types.

Training each model probably takes only a few (3-4) hours on a GPU.

The models and logs are saved to the directory at `OPENAI_LOGDIR`. If you want to save the model files to your desired
folder, modify the varibale via `export OPENAI_LOGDIR=...`

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

I'll add this soon!

## To-do List

* Release pretrained models on AFHQ and yosemite datasets
* Add color translation experiments
* Add scripts to translate between AFHQ, yosemite images