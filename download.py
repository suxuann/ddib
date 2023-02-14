import argparse
import os
import urllib
from pathlib import Path

import gdown


class Constant:
    SyntheticURL = "https://drive.google.com/drive/folders/1YRP6nt96OJUOzEYY6N_Qh5xb3wEVFSjg"
    ImageNetDatasetURL = "https://drive.google.com/drive/folders/17vIa5bMNYql9IbSCsbQ8ECGpYuLhg6KF"
    ImageNetClassifier = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt"
    ImageNetDiffusion = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt"


class Experiment:
    Synthetic = "synthetic"
    ImageNet = "imagenet"
    ImageNetDataset = "imagenet_dataset"
    All = [Synthetic, ImageNet, ImageNetDataset]


def download_synthetic_checkpoints():
    """
    Creates a new folder under `models/synthetic`. Then, downloads the synthetic
    model checkpoints into the folder.
    """
    path = "./models/synthetic"
    Path(path).mkdir(parents=True, exist_ok=True)
    gdown.download_folder(Constant.SyntheticURL, output=path,
                          quiet=False, use_cookies=False)


def download_imagenet_checkpoints():
    path = "./models/imagenet"
    Path(path).mkdir(parents=True, exist_ok=True)
    for file in [Constant.ImageNetClassifier, Constant.ImageNetDiffusion]:
        print(f"Downloading model weights: {file}")
        filepath = os.path.join(path, os.path.basename(file))
        print(f"    Saving to {filepath}")
        urllib.request.urlretrieve(file, filename=filepath)


def download_imagenet_dataset():
    """Downloads a subset of ImageNet validation set."""
    path = "./models/imagenet_dataset"
    Path(path).mkdir(parents=True, exist_ok=True)
    gdown.download_folder(Constant.ImageNetDatasetURL, output=path,
                          quiet=False, use_cookies=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        default="",
        help="Which experiment in the DDIBs paper to run."
    )
    args = parser.parse_args()

    exp = args.exp
    assert args.exp.lower() in Experiment.All

    print(f"Configuring the project for {exp} experiments.")
    if exp == Experiment.Synthetic:
        download_synthetic_checkpoints()
    elif exp == Experiment.ImageNet:
        download_imagenet_checkpoints()
    elif exp == Experiment.ImageNetDataset:
        download_imagenet_dataset()




if __name__ == "__main__":
    main()
