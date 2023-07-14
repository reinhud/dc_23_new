import os

import numpy as np
from timm.data import create_transform
from timm.data.dataset import ImageDataset
from timm.utils.misc import natural_key

from src.data.datasets.coin_data_reader import CoinDataReader


class CoinData:
    types = ".jpg"

    def __init__(
        self,
        folder_path: str = "src/data/raw/CN_dataset_04_23/data_types_example",
    ) -> None:
        self.folder_path = folder_path
        self.images_and_targets, self.class_to_idx = self._find_images_and_targets()
        if len(self.images_and_targets) == 0:
            raise RuntimeError(
                f"Found 0 images in subfolders of {folder_path}. "
                f'Supported image extensions are {", ".join(self.types)}'
            )
        self.num_classes = len(self.class_to_idx)

    def _find_images_and_targets(self, sort: bool = True):
        """Walk folder to discover images and map them to classes by folder names.
        Args:
            folder: root of folder to recrusively search
            sort: re-sort found images by name (for consistent ordering)

        Returns:
            A list of image and target tuples, class_to_idx mapping
        """
        labels = []
        filenames = []
        for root, subdirs, files in os.walk(
            self.folder_path, topdown=False, followlinks=True
        ):
            rel_path = (
                os.path.relpath(root, self.folder_path)
                if (root != self.folder_path)
                else ""
            )
            label = rel_path.replace(os.path.sep, "_")
            for f in files:
                base, ext = os.path.splitext(f)
                if ext.lower() in self.types:
                    filenames.append(os.path.join(root, f))
                    labels.append(label)
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
        images_and_targets = [
            (f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx
        ]
        if sort:
            images_and_targets = sorted(
                images_and_targets, key=lambda k: natural_key(k[0])
            )
        return images_and_targets, class_to_idx

    def _generate_reader(self, index: list):
        # find the correct images and targets by index
        images_and_targets_reader = [self.images_and_targets[i] for i in index]
        # create a reader
        return CoinDataReader(self.folder_path, images_and_targets_reader)

    def _split_train_val(
        self,
        val_pct: float = 0.3,  # TODO: dont hardcode
        shuffle: bool = True,
        random_seed: int = 42,
    ):
        # get the number of images
        dataset_size = len(self.images_and_targets)
        indices = list(range(dataset_size))
        split = int(np.floor(val_pct * dataset_size))
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_index, val_index = indices[split:], indices[:split]
        return train_index, val_index

    def _generate_train_val_reader(self, val_pct: float, shuffle: bool = True):
        # split the data into train and val
        train_index, val_index = self._split_train_val(val_pct, shuffle)
        # create readers for train and val
        train_reader = self._generate_reader(train_index)
        val_reader = self._generate_reader(val_index)
        return train_reader, val_reader

    def generate_train_val_datasets(
        self,
        val_pct: float,
        image_size,
        data_mean,
        data_std,
        shuffle: bool = True,
    ):
        # create transforms for train and val
        train_transforms = create_transform(
            input_size=image_size,
            is_training=True,
            mean=data_mean,
            std=data_std,
            auto_augment="rand-m7-mstd0.5-inc1",    # TODO: load from env, what does this eman?
        )

        eval_transforms = create_transform(
            input_size=image_size, mean=data_mean, std=data_std
        )
        # create readers for train and val
        train_reader, val_reader = self._generate_train_val_reader(val_pct, shuffle)
        # create datasets
        train_dataset = ImageDataset(
            root=self.folder_path, reader=train_reader, transform=train_transforms
        )
        val_dataset = ImageDataset(
            root=self.folder_path, reader=val_reader, transform=eval_transforms
        )
        return train_dataset, val_dataset