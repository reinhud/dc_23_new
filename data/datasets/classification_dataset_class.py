"""Implementation of a custom dataset class in pytorch."""
import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torchvision import transforms
from enum import Enum
from pathlib import Path
from PIL import Image
from typing import Tuple
import numpy as np

from data.datasets.helper.classification_data_helper import find_classes_in_folder


class CoinData(Enum):
    """Enum for the differnt paths of coin datasets.
    
    The datasets are structured so each folder name in the given paths is the relavative class for the images within that folder.
    """
    TYPES_EXAMPLE: str = "/workspaces/data-challenge-sose23/src/data/raw/CN_dataset_04_23/data_types_example/"
    COINS: str = "/workspaces/data-challenge-sose23/src/data/raw/CN_dataset_04_23/dataset_coins/"
    MINTS: str = "/workspaces/data-challenge-sose23/src/data/raw/CN_dataset_04_23/dataset_mints/"
    TYPES: str = "/workspaces/data-challenge-sose23/src/data/raw/CN_dataset_04_23/dataset_types/"


class ClassificationDataset(Dataset):

    def __init__(self, data_dir: CoinData = CoinData.TYPES_EXAMPLE, transform: transforms.Compose = None) -> None:
        
        # Get all image paths
        self.paths = list(Path(data_dir.value).glob("*/*.jpg"))
        
        # Setup transforms
        self.transform = transform

        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes_in_folder(data_dir.value)

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]

        return Image.open(image_path) 

    # Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."

        return len(self.paths)
    
    # Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:

            return self.transform(img), class_idx # return data, label (X, y)
        else:

            return img, class_idx # return data, label (X, y)
        
    def get_class_distribution(self):
        """Returns a dictionary with class names and their distribution in the dataset."""
        class_distribution = {class_name: 0 for class_name in self.classes}
        for path in self.paths:
            class_name = path.parent.name
            class_distribution[class_name] += 1

        return class_distribution
  
    def create_random_dataloaders(self, batch_size: int = 32, validation_split: float = 0.2, transform: transforms.Compose = None, shuffle_dataset: bool = True, random_seed: int = 42) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Returns randomly sampled training and validation dataloaders."""
        # Creating data indices for training and validation splits:
        dataset_size = len(self.paths)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating Pytorch data loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # Create dataloaders
        train_loader = DataLoader(
            self,
            batch_size=batch_size, 
            sampler=train_sampler
            )
        validation_loader = DataLoader(
            dataset=self, 
            batch_size=batch_size, 
            sampler=valid_sampler
            )
        
        return train_loader, validation_loader
    
    def create_weighted_dataloaders(self, batch_size: int = 32, validation_split: float = 0.2, transform: transforms.Compose = None, shuffle_dataset: bool = True, random_seed: int = 42) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Returns weighted training and validation dataloaders.
        
        This helps with class imbalance, increasing sampling from minority classes making sure
        that each batch has the same number of samples from each class.
        """
        # list of traget classes
        class_list = torch.tensor(self.classes)

        # class counts


