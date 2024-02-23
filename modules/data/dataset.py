import os
import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List
from modules.utils.labels import labels as cityscapes_labels


@dataclass
class CityScapesDatasetConfig:
        """
        The configuration for the CityScapesDataset

        Args:
            images_dir (str): The directory containing the images
            labels_dir (str): The directory containing the labels
            dataset_category (str): The category of the dataset (train, val, test)
            image_size (tuple): The size to resize the images and labels to
            transform (transforms): The transform to apply to the images and labels
            decode_masks (bool): Whether to decode the masks to RGB images
            ignore_index (int): The index to ignore in the masks
            ignore_void_classes (bool): Whether to ignore void classes
            ignore_labels (List): The list of labels to ignore
        """
        images_dir: str
        labels_dir: str
        dataset_category: str = "train"
        image_size: tuple = (1024, 2048)
        transform: transforms = None
        decode_masks: bool = False
        ignore_index: int = None
        ignore_void_classes: bool = False
        ignore_labels: List = None



class CityScapesDataset(Dataset):

    """
    The CityScapes Dataset class.
    """
    
    def __init__(
        self,
        config: CityScapesDatasetConfig,
    ):
        """
        Instantiates an object of the CityScapesDataset class.

        Args:
            config (CityScapesDatasetConfig): The configuration for the dataset.
        """
        self.config = config
        self.images_path = os.path.join(self.config.images_dir, self.config.dataset_category)
        self.labels_path = os.path.join(self.config.labels_dir, self.config.dataset_category)
        self.images = self.recursive_glob(self.images_path)
        self.label_colours = {label.id: torch.tensor(label.color) for label in cityscapes_labels}
        self.label_ids = {label.id:label.trainId for label in cityscapes_labels}
        self.label_names = {label.id:label.name for label in cityscapes_labels}
        self.ignore_labels = self.config.ignore_labels if self.config.ignore_labels is not None else []

        # Get the class labels to ignore before getting the valid class labels to keep
        void_label_ids = {-1}
        if self.config.ignore_void_classes:
            void_label_ids.update({label.id for label in cityscapes_labels if label.trainId == 255})
        if len(self.ignore_labels) > 0:
            void_label_ids.update({label.id for label in cityscapes_labels if label.id in self.ignore_labels})
        valid_label_ids = {label.id for label in cityscapes_labels if label.id not in void_label_ids}
        self.num_classes = len(valid_label_ids)
        self.ignore_index = self.num_classes  if self.config.ignore_index is None else self.config.ignore_index
        self.void_label_ids = {old_id:self.ignore_index for old_id in void_label_ids}
        self.valid_label_ids = {old_id:new_id for (old_id, new_id) in zip(valid_label_ids, range(self.num_classes))}
        self.valid_label_colours = {self.valid_label_ids[label_id]: self.label_colours[label_id] for label_id in valid_label_ids}
        self.void_label_names = {label_id: self.label_names[label_id] for label_id in void_label_ids}
        self.valid_label_names = {label_id: self.label_names[label_id] for label_id in valid_label_ids}

    def set_config(self, config):
        """
        Change the datasets confoguration
        """
        self.config = config
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index].rstrip()
        label_path = os.path.join(
            self.labels_path,
            image_path.split(os.sep)[-2],
            os.path.basename(image_path)[:-15] + "gtFine_labelIds.png",
        )

        image, mask = self.read_data(image_path, label_path)
        mask = self.re_encode_mask(mask)
        if self.config.transform is not None:
            image, mask = self.config.transform(image, mask)

        return image, mask

    def re_encode_mask(self, mask):      
        """
        Re-encode the masks to match valid label ids and encodes ignore index
        """        
        for label_ids in self.void_label_ids, self.valid_label_ids:
            for old_label_id, new_label_id in label_ids.items():
                mask[mask == old_label_id] = new_label_id

        return mask

    def decode_mask(self, mask):
        """
        Converts label masks to their equivalent RGB masks
        """
        rgb_mask = torch.ones(self.config.image_size + (3,)) * 255
        rgb_mask = rgb_mask.type(mask.dtype)

        for _, new_label_id in self.valid_label_ids.items():
            rgb_mask[mask == new_label_id] = self.valid_label_colours[new_label_id]

        return rgb_mask / 255.0

    def read_data(self, image_path, label_path):
        image = transforms.functional.pil_to_tensor(Image.open(image_path))
        image = image / 255 if image.max() > 1 else image 
        mask = torch.as_tensor(np.array(Image.open(label_path)), dtype=torch.int64)
        
        H, W = self.config.image_size
        if (image.shape[0] != H) or (image.shape[1] != W):
            interpolation = transforms.InterpolationMode.NEAREST_EXACT 
            if W > image.shape[1]:
                interpolation = transforms.InterpolationMode.BICUBIC
            resize = transforms.Resize(size=self.config.image_size, interpolation=interpolation)
            image = resize(image)
            mask = resize(mask.unsqueeze(0)).squeeze()
        
        if self.config.decode_masks:
            mask = self.decode_mask(mask)

        return image, mask
    
    @property
    def dataset_info(self):
        return {
            "Number of Images": len(self.images),
            "Image Size": self.config.image_size, 
            "Number of Classes": self.num_classes,
            "Void Class Label IDs (old_id: new_id)": self.valid_label_ids,
            "Valid Class Label IDs (old_id: new_id)": self.valid_label_ids,
        }
        
        
    @staticmethod
    def recursive_glob(rootdir=".", suffices={".png", ".jpg"}):
        data = []
        for suffix in suffices:
            data.extend([
            os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames
            if filename.endswith(suffix)
        ])

        return data

if __name__ == "__main__":
    train_data_config = CityScapesDatasetConfig(
        images_dir = "datasets/images/leftImg8bit/", 
        labels_dir = "datasets/labels/gtFine", 
        image_size = (512,1024),
        dataset_category = "train",
    )
    train_dataset = CityScapesDataset(train_data_config)
    print(f"Valid Labels: {train_dataset.valid_label_ids}")
    print(f"Void Labels: {train_dataset.valid_label_ids}")
