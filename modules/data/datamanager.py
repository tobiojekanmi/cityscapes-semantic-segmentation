import os
from dataclasses import dataclass
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Optional, List, Callable
from .dataset import CityScapesDataset, CityScapesDatasetConfig


@dataclass
class CityScapesDataManagerConfig:
    """
    The configuration for the CityScapesDataManager

    Args:
        images_dir (str): The directory containing the images
        labels_dir (str): The directory containing the labels
        image_size (tuple): The size to resize the images and labels to
        ignore_void_classes (bool): Whether to ignore void classes
        ignore_index (int): The index to ignore in the masks
        ignore_labels (List): The list of labels to ignore
        train_transforms (transforms): The transform to apply to the images and labels for training
        valid_transforms (transforms): The transform to apply to the images and labels for validation
        test_transforms (transforms): The transform to apply to the images and labels for testing
        decode_train_masks (bool): Whether to decode the masks to RGB images for training
        decode_valid_masks (bool): Whether to decode the masks to RGB images for validation
        decode_test_masks (bool): Whether to decode the masks to RGB images for testing
        batch_size (int): The batch size for the dataloader
        num_workers (int): The number of workers for the dataloader
        pin_memory (bool): Whether to pin memory for the dataloader
        shuffle (bool): Whether to shuffle the dataloader data output
        collate_fn (Callable): The collate function for the dataloader
        drop_last (bool): Whether to drop the last batch of the dataloader
    """
    images_dir: str
    labels_dir: str
    image_size: tuple = (1024, 2048)
    ignore_void_classes: bool = True
    ignore_index: Optional[int] = None
    ignore_labels: Optional[List[int]] = None
    train_transforms: Optional[transforms.Compose] = None
    valid_transforms: Optional[transforms.Compose] = None
    test_transforms: Optional[transforms.Compose] = None
    decode_train_masks: Optional[bool] = False
    decode_valid_masks: Optional[bool] = False
    decode_test_masks: Optional[bool] = False
    batch_size: int = 16
    num_workers: Optional[int] = 2
    pin_memory: Optional[bool] = False
    shuffle: Optional[bool] = True
    collate_fn: Optional[Callable] = None
    drop_last: Optional[bool] = False



class CityScapesDataManager:
    """
    The CityScapesDataManager class.
    """

    def __init__(
            self,
            config: CityScapesDataManagerConfig,
        ):
        """
        Instantiates an object of the CityScapesDataManager class.

        Args:
            config (CityScapesDataManagerConfig): The configuration for the data manager.
        """
        self.config = config
    
    @property
    def train_dataset_config(self):
        if not os.path.exists(os.path.join(self.config.images_dir, "train")) or \
            not os.path.exists(os.path.join(self.config.labels_dir, "train")):
            return None
        return CityScapesDatasetConfig(
            images_dir = self.config.images_dir, 
            labels_dir = self.config.labels_dir, 
            dataset_category = "train",
            image_size = self.config.image_size,
            transform = self.config.train_transforms,
            decode_masks = self.config.decode_train_masks,
            ignore_index = self.config.ignore_index,
            ignore_void_classes = self.config.ignore_void_classes,
            )

    @property
    def valid_dataset_config(self):
        if not os.path.exists(os.path.join(self.config.images_dir, "val")) or \
            not os.path.exists(os.path.join(self.config.labels_dir, "val")):
            return None
        return CityScapesDatasetConfig(
            images_dir = self.config.images_dir, 
            labels_dir = self.config.labels_dir, 
            dataset_category = "val",
            image_size = self.config.image_size,
            transform = self.config.valid_transforms,
            decode_masks = self.config.decode_valid_masks,
            ignore_index = self.config.ignore_index,
            ignore_void_classes = self.config.ignore_void_classes,
            )
    
    @property
    def test_dataset_config(self):
        if not os.path.exists(os.path.join(self.config.images_dir, "test")) or \
            not os.path.exists(os.path.join(self.config.labels_dir, "test")):
            return None
        return CityScapesDatasetConfig(
            images_dir = self.config.images_dir, 
            labels_dir = self.config.labels_dir, 
            dataset_category = "test",
            image_size = self.config.image_size,
            transform = self.config.test_transforms,
            decode_masks = self.config.decode_test_masks,
            ignore_index = self.config.ignore_index,
            ignore_void_classes = self.config.ignore_void_classes,
            )
    
    def get_train_dataset(self):
        if self.train_dataset_config is None:
            expected_paths = (
                os.path.join(self.config.images_dir, "train"),
                os.path.join(self.config.labels_dir, "train"),
                )
            raise ValueError(f"One of {expected_paths} does not exist!!")
        return CityScapesDataset(self.train_dataset_config)

    def get_valid_dataset(self):
        if self.valid_dataset_config is None:
            expected_paths = (
                os.path.join(self.config.images_dir, "valid"),
                os.path.join(self.config.labels_dir, "valid"),
                )
            raise ValueError(f"One of {expected_paths} does not exist!!")
        return CityScapesDataset(self.valid_dataset_config)

    def get_test_dataset(self):
        if self.test_dataset_config is None:
            expected_paths = (
                os.path.join(self.config.images_dir, "test"),
                os.path.join(self.config.labels_dir, "test"),
                )
            raise ValueError(f"One of {expected_paths} does not exist!!")
        return CityScapesDataset(self.test_dataset_config)

    def get_train_dataloader(self):
        return DataLoader(
            self.get_train_dataset(),
            batch_size = self.config.batch_size,
            shuffle = self.config.shuffle,
            num_workers = self.config.num_workers,
            pin_memory = self.config.pin_memory,
            collate_fn = self.config.collate_fn,
            drop_last = self.config.drop_last,
            )

    def get_valid_dataloader(self):
        return DataLoader(
            self.get_valid_dataset(),
            batch_size = self.config.batch_size,
            num_workers = self.config.num_workers,
            pin_memory = self.config.pin_memory,
            collate_fn = self.config.collate_fn,
            drop_last = self.config.drop_last,
            )

    def get_test_dataloader(self):
        return DataLoader(
            self.get_test_dataset(),
            batch_size = self.config.batch_size,
            num_workers = self.config.num_workers,
            pin_memory = self.config.pin_memory,
            collate_fn = self.config.collate_fn,
            drop_last = self.config.drop_last,
            )

    def decode_mask(self, mask):
        return self.get_train_dataset().decode_mask(mask)

    @property
    def num_classes(self):
        return self.get_train_dataset().num_classes

    @property
    def ignore_index(self):
        return self.get_train_dataset().ignore_index

    @property
    def void_label_ids(self):
        return self.get_train_dataset().void_label_ids

    @property
    def valid_label_ids(self):
        return self.get_train_dataset().valid_label_ids

    @property
    def label_names(self):
        return self.get_train_dataset().label_names

    @property
    def ignore_labels(self):
        return self.get_train_dataset().ignore_labels

    @property
    def train_dataset_info(self):
        return self.get_train_dataset().dataset_info
        
    @property
    def valid_dataset_info(self):
        return self.get_valid_dataset().dataset_info
        
    @property
    def test_dataset_info(self):
        return self.get_valid_dataset().dataset_info





if __name__ == "__main__":
    datamanager_config = CityScapesDataManagerConfig(
        images_dir = "datasets/images/leftImg8bit/", 
        labels_dir = "datasets/labels/gtFine", 
        image_size = (1024, 2048),
        batch_size = 2,
        shuffle = True
    )
    datamanager = CityScapesDataManager(datamanager_config)
    train_dataloader = datamanager.get_train_dataloader()
    valid_dataloader = datamanager.get_valid_dataloader()
    train_batch = next(iter(train_dataloader))
    valid_batch = next(iter(valid_dataloader))

    print(train_batch[0].shape, train_batch[1].shape)
    print(valid_batch[0].shape, valid_batch[1].shape)
