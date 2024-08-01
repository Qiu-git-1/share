import os

import load_mvtec_loco as mvt
import argparse
import pathlib
import torch
import numpy as np
import torch.nn.functional as F
'''
The main purpose of this Python code is to preprocess the image from the MVTec Loco dataset and save it as a.npy file.
The MVTec Loco dataset contains images of multiple products (such as coffee cups, bottles, etc.) and is divided into normal and abnormal categories.
The exception category is further divided into logical anomalies and structural anomalies.

The execution process of the code is roughly as follows:
1. Import the necessary libraries and set up the command line parameter parser.
2. Define the data set storage path and the basic parameters for each object type.
3. For each object type, traverse the data of its logical exception and structural exception types.
4. Load training set, verification set and test set data.
5. Initialize the corresponding array to store the preprocessed data.
6. Go through the data loader, load the image data into the initialized array, and adjust the image aspect ratio according to the maximum side length to ensure that it matches the set im_size.
7. Use the bilinear interpolation method F.i TERPOLate to adjust the size of the image and ensure that the aspect ratio of the original image is preserved.
8. Save the preprocessed data as.npy files for subsequent model training and evaluation.
The code uses the load_mvtec_loco module to load the MVTec Loco dataset, which defines the detailed flow of data loading.
The code also includes functions for path operations and array operations, such as the resize_array function to resize the array.
'''
def resize_array(new_img_size, in_array):
    # Creates a group of zeros with the same shape as the input array
    array_new = torch.zeros(in_array.shape)
    # Interpolates the input array to resize it to the specified new size
    array_interp = F.interpolate(in_array, (int(new_img_size[0]), int(new_img_size[1])))
    # Copies the interpolated array into a new set of zeros, keeping the data in the original size portion
    array_new[:, :, :int(new_img_size[0]), :int(new_img_size[1])] = array_interp
    return array_new

# Gets the parent path of the current script file
parent_path = pathlib.Path(__file__).parent.absolute()
print("parent_path",parent_path)
# Set up the command line argument parser
parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--im_size_before_crop', default=1024, type=int)
parser.add_argument('--im_size', default=1024, type=int)
args = parser.parse_args()
# Data set storage path
data_matrices_path = "./dataset_loco/data_matrices"
# Processing for each object type
for mvtype in (['breakfast_box','juice_bottle','pushpins','screw_bag','splicing_connectors']):
    print(mvtype)

    if "pushpins" in mvtype:
        img_aspects = [1700,1000]
    if "screw_bag" in mvtype:
        img_aspects = [1600,1100]
    if "splicing_connectors" in mvtype:
        img_aspects = [1700,850]
    if "juice_bottle" in mvtype:
        img_aspects = [800,1600]
    if "breakfast_box" in mvtype:
        img_aspects = [1600,1280]
    if "breakfast_box_short" in mvtype:
        img_aspects = [1600,1280]
    if "pushpins_short" in mvtype:
        img_aspects = [1700,1000]
    if "splicing_connectors_short" in mvtype:
        img_aspects = [1700,850]
    # Adjust the aspect ratio to ensure that the large edge matches the set im_size
    img_aspects = [img_aspects[1], img_aspects[0]]
    aspect_large_side = np.max(img_aspects)
    size_ratio = aspect_large_side/args.im_size
    img_aspects = img_aspects/size_ratio
    # Iterate over two types of exceptions: logical exception and structural exception
    for anom_type in ['logical_anomalies','structural_anomalies']:

        if anom_type == "logical_anomalies":
            name_suffix = "loco"
        if anom_type == "structural_anomalies":
            name_suffix = "struct"
        # The output path is set to a subdirectory under the data set path
        out_path = os.path.join(data_matrices_path, "%s_%s"%(mvtype, name_suffix))
        os.makedirs(out_path, exist_ok=True)
        # Data sets are loaded according to different training phases
        #mvtype：Level 1 directory in the data set   anom_type：Exception type   args.im_size：Picture size   args.im_size_before_crop：Picture cropping size
        trainloader = mvt.get_mvt_loader(parent_path, 'train', mvtype, anom_type, args.im_size, args.im_size_before_crop, is_loco = True)
        validloader = mvt.get_mvt_loader(parent_path, 'validation', mvtype, anom_type, args.im_size, args.im_size_before_crop, is_loco = True)
        testloader = mvt.get_mvt_loader(parent_path, 'test', mvtype, anom_type, args.im_size, args.im_size_before_crop, is_loco = True)


        # Initializes an empty array of training, validation, and test data
        list_trainloader = torch.zeros((len(trainloader),3, args.im_size, args.im_size))
        print("Resized list_trainloader shape:", list_trainloader.shape)
        list_validloader = torch.zeros((len(validloader),3, args.im_size, args.im_size))
        list_testloader = torch.zeros((len(testloader),3, args.im_size, args.im_size))
        # Initializes the image level label array
        image_level_label = np.zeros(len(testloader))

        # Load the training set data into the appropriate array
        for batch_idx, (img, label) in enumerate(trainloader):
            list_trainloader[batch_idx] = (img)
        # Load the validation set data into the appropriate array
        for batch_idx, (img, label) in enumerate(validloader):
            list_validloader[batch_idx] = (img)
        # Load the test set data into the appropriate array and record the image-level label
        for batch_idx, (img, label) in enumerate(testloader):
            list_testloader[batch_idx] = (img)
            image_level_label[batch_idx] = label
        # Resize the loaded training, validation, and test data
        list_trainloader = resize_array(img_aspects, list_trainloader)
        list_validloader = resize_array(img_aspects, list_validloader)
        list_testloader = resize_array(img_aspects, list_testloader)

        # Print the adjusted training data array shape
        print("list_trainloader",list_trainloader.shape)

        # Save the processed data as a.npy file
        np.save(os.path.join(out_path, "train_data.npy"),list_trainloader.numpy())
        np.save(os.path.join(out_path, "valid_data.npy"),list_validloader.numpy())
        np.save(os.path.join(out_path, "test_data.npy"),list_testloader.numpy())
        np.save(os.path.join(out_path, "image_level_label.npy"),image_level_label)
