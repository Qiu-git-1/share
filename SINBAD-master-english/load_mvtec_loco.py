import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import PIL.Image as Image
import os
'''
For loading and preprocessing image datasets, specially designed for MVTec AD datasets and LOCOS datasets. It provides two ways to load data sets,
Irrelevant test set samples can be filtered out as needed.

1.Dependency import: The script begins by importing some necessary PyTorch and other libraries for data processing and loading.

2.Auxiliary functions:
- default_loader: default function for loading images.
-find_classes: Finds and sorts categories in the directory.

3.Custom Dataset Class MyDataset:
- This class inherits from torch.utils.data.Dataset and is used to load and preprocess image data.
- In the initialization function __init__, it sets up the data conversion, selects the correct data set path, and loads the data according to the parameters.
It also implements __getitem__ and __len__ methods to make it easy to access data by index and get data set length.

4.Data loading function get_mvt_loader:
- this function creates a MyDataset instance, with the torch. Utils. Data. The DataLoader encapsulated, so that the bulk loading and data processing.
This script can be used as a basis for loading MVTec AD or LOCOS datasets in other PyTorch projects.
'''

def default_loader(path):
    """Default image loading function."""
    return Image.open(path).convert('RGB')


def find_classes(dir):
    """Find and sort the categories in the directory."""
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class MyDataset(Dataset):
    """A custom dataset class for loading and preprocessing image data."""

    def __init__(self, parent_path: str, which_set: str, class_name: str, anom_type: str, 
                 img_size: int = 1024, img_resize: int = 256, is_loco: bool = True):
        """Data set initialization function."""
        transform = transforms.Compose([
            transforms.Resize((img_resize, img_resize)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])

        # Select the correct data set path
        data_path = 'dataset_loco' if is_loco else 'dataset_mvtec'

        # Set the correct folder path according to the parameters
        if which_set == 'train':
            fold_path = os.path.join(parent_path, data_path, class_name, "train")
        elif which_set == 'validation':
            fold_path = os.path.join(parent_path, data_path, class_name, "validation")
        elif which_set == 'test':
            fold_path = os.path.join(parent_path, data_path, class_name, "test")
            print(f"parent_path:{parent_path}, data_path:{data_path}, class_name:{class_name}")

        # Create a dataset instance
        dataset = torchvision.datasets.ImageFolder(fold_path, transform)

        # Create a DataLoader
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                              shuffle=False, num_workers=0, drop_last=False)

        # Initialization variable
        target_list = np.zeros(len(trainloader))
        imgs = torch.zeros((len(trainloader), 3, img_size, img_size))
        label_list = torch.zeros((len(trainloader)))
        is_relevant_list = np.zeros((len(trainloader)))

        # Traverse the DataLoader, loading and preprocessing the data
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # Determine whether the test set samples are related
            if which_set == 'test':
                if (os.path.join('/test/', anom_type) in dataset.imgs[batch_idx][0] or
                        '/test/good' in dataset.imgs[batch_idx][0] or anom_type == 'all'):
                    is_relevant_list[batch_idx] = 1
                else:
                    is_relevant_list[batch_idx] = 0
            else:
                is_relevant_list[batch_idx] = 1

            # Set label
            if which_set == 'test' and int(dataset.imgs[batch_idx][1]) > 0:
                label = 1
            else:
                label = 0

            # Store data
            imgs[batch_idx] = inputs[0]
            target_list[batch_idx] = targets.item()
            label_list[batch_idx] = label

        # Filter out irrelevant samples
        relevant_inds = np.where(is_relevant_list == 1)
        self.targets = np.array(label_list)[relevant_inds]
        self.imgs = imgs[relevant_inds]

    def __getitem__(self, index):
        """Gets data and labels for the specified index."""
        img = self.imgs[index]
        label = self.targets[index]
        return img, label

    def __len__(self):
        """Gets the length of the data set."""
        return len(self.imgs)


def get_mvt_loader(parent_path: str, which_set: str = 'train', class_name: str = "breakfast_box", 
                   anom_type: str = "logical_anomalies", img_size: int = 1024, img_resize: int = 1024, 
                   is_loco: bool = True):
    """Creates and returns a data loader."""
    mvt_data_in = MyDataset(parent_path, which_set, class_name, anom_type, img_size, img_resize, is_loco=is_loco)
    mvt_loader = torch.utils.data.DataLoader(
        mvt_data_in,
        batch_size=1, shuffle=False,
        num_workers=0)
    return mvt_loader