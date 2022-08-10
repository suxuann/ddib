import blobfile as bf
import math
import numpy as np
import random
from PIL import Image
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset


def load_data(
        *,
        data_dir,
        batch_size,
        image_size,
        class_cond=False,
        deterministic=False,
        random_crop=False,
        random_flip=True,
        in_channels=3
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    :param in_channels: new parameter in DDIBs as we experimented with grayscale
                        images
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
        in_channels=in_channels
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def load_source_data_for_domain_translation(
        *,
        data_dir,
        batch_size,
        image_size,
        label,
        class_cond=False,
        in_channels=3,
        file_filter=None,
        retain_zero_class=False
):
    """
    New in DDIBs: Loads the source dataset for domain translation.
    For the dataset, create a generator over (images, kwargs) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair. Defaults to 1.
    :param image_size: the size to which images are resized.
    :param label: include a "y" key in returned dicts for class label.
    :param file_filter: a filter to apply on all_files if not None
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = list_image_files(data_dir, retain_zero_class=retain_zero_class)
    if file_filter is not None:
        all_files = [x for x in all_files if file_filter(x)]
    classes = [label for _ in all_files]  # TODO This may have bugs
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        in_channels=in_channels,
        random_flip=False,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    yield from loader


def list_image_files(data_dir, retain_zero_class=False):
    """List images files in the directory (not recursively)."""
    files = sorted([file.zfill(20) for file in bf.listdir(data_dir)])
    results = []
    for entry in files:
        entry = entry.lstrip('0')
        if retain_zero_class and entry.startswith("_"):
            entry = f"0{entry}"
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
    return results


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
            self,
            resolution,
            image_paths,
            classes=None,
            shard=0,
            num_shards=1,
            random_crop=False,
            random_flip=True,
            in_channels=3
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.in_channels = in_channels

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        if self.in_channels == 3:
            pil_image = pil_image.convert("RGB")
        else:
            pil_image = pil_image.convert("L")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)
        if len(arr.shape) < 3:
            arr = arr[:, :, np.newaxis]  # Adds a single channel

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = dict()
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def get_image_filenames_for_label(label):
    """
    Returns the validation files for images with the given label. This is a utility
    function for ImageNet translation experiments.
    :param label: an integer in 0-1000
    """
    # First, retrieve the synset word corresponding to the given label
    synsets_filepath = "./../evaluations/synset_words.txt"
    synsets = [line.split()[0] for line in open(synsets_filepath).readlines()]
    synset_word_for_label = synsets[label]

    # Next, build the synset to ID mapping
    synset_mapping_filepath = "./../evaluations/map_clsloc.txt"
    synset_to_id = dict()
    with open(synset_mapping_filepath) as file:
        for line in file:
            synset, class_id, _ = line.split()
            synset_to_id[synset.strip()] = int(class_id.strip())
    true_label = synset_to_id[synset_word_for_label]

    # Finally, return image files corresponding to the true label
    validation_ground_truth_filepath = "./../evaluations/ILSVRC2012_validation_ground_truth.txt"
    source_data_labels = [int(line.strip()) for line in open(validation_ground_truth_filepath).readlines()]
    image_indexes = [i + 1 for i in range(len(source_data_labels)) if true_label == source_data_labels[i]]
    output = [f"ILSVRC2012_val_{str(i).zfill(8)}.JPEG" for i in image_indexes]
    return output
