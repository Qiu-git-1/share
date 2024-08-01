import os
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score
import argparse
from scipy.stats import trim_mean


'''
It is used to calculate the AUC (Area Under the ROC Curve) value in multivariate time series data (MVTS) anomaly detection task.

It basically does the following things:
1. Load anomaly maps with different resolutions and repetitions.
2. Scale the exception maps of different resolutions to the same size (in this case, 224x224).
3. Calculate the AUC value of each pixel or the entire image.
4. Calculate the final AUC value by combining anomaly mappings of different resolutions and sizes.
5. Output the AUC value and average AUC value of each type of anomaly detection task.

The key functions and flows in the code are as follows:
- get_anom_map_224: Load and handle 224x224 resolution exception mappings.
- get_anom_map: Load and process exception maps of different resolutions and scale them to the same size.
- Loop through each exception type, calculating the AUC value for each pixel.
- The final AUC value is calculated by combining the anomaly mappings of different resolutions.
The argparse library is used in the code to parse command line parameters, such as the type of exception detection task (mvtype), the result version path (version), and the result version path (version).
Number of repetitions (rep_num_224) and weight factor (lambda_factor)
'''

parser = argparse.ArgumentParser(description='PyTorch Seen Testing Category Training')
parser.add_argument('--mvtype', default='breakfast_box_loco', type=str) #mvtec class / loco / struct / all
parser.add_argument('--version', default='/path/to/sinbad_runs/results/ver1_pyramid_lvl_#', type=str)
parser.add_argument('--version_224', default='/path/to/sinbad_runs/results/ver1_pyramid_lvl_#', type=str)
parser.add_argument('--rep_num_224', default=100, type=int)
parser.add_argument('--lambda_factor', default=0.1, type=float)

args = parser.parse_args()

fold_path = "/root/autodl-tmp/"#The current project path needs to be changed

if args.mvtype == "all":
    mvtype_list = np.array(['breakfast_box_loco','juice_bottle_loco','pushpins_loco','screw_bag_loco','splicing_connectors_loco',
                                'breakfast_box_struct','juice_bottle_struct','pushpins_struct','screw_bag_struct','splicing_connectors_struct'])
elif args.mvtype == "loco":
    mvtype_list = np.array(['breakfast_box_loco','juice_bottle_loco','pushpins_loco','screw_bag_loco','splicing_connectors_loco'])
elif args.mvtype == "struct":
    mvtype_list = np.array(['breakfast_box_struct','juice_bottle_struct','pushpins_struct','screw_bag_struct','splicing_connectors_struct'])
else:
    mvtype_list = [args.mvtype]

reg_eps = 0.00001


roc_mean_list = []
roc_max_list = []
roc_loc_list = []
roc_clr_list = []
roc_mean_w_clr_list = []
roc_max_w_clr_list = []

mvtypes_roc = []



def get_anom_map_224(resized_anom_maps_res_lvls, set_string = "anom_map_test"):
    #Responsible for loading and preprocessing 224x224 resolution exception maps
    anom_maps = []
    lvl_version = args.version_224.replace("#", "%d"%(224))
    res_fold_temp = os.path.join(fold_path, lvl_version + "/rep_num_#", mvtype)
    # Go through the number of times each repetition
    for j in range(0,args.rep_num_224): #range(args.num_of_224_reps):
        res_fold = res_fold_temp.replace("#", "%d"%(j))
        # Iterate over the file names in the folder
        for file_name in (os.listdir(res_fold)):
            file_path = os.path.join(res_fold, file_name)
            # If the file name contains "anom_maps", the array is loaded and processed
            if ("anom_maps" in file_name):

                array = np.load(file_path)
                array = array[set_string]
                anom_maps.append(array)
    # Converts the exception mapping array to a NumPy array and returns
    anom_maps = np.array(anom_maps)
    return anom_maps[:,0]

def get_anom_map(resized_anom_maps_res_lvls, set_string = "anom_map_test", rep_num = 0):
   #Responsible for loading and preprocessing exception maps of different resolutions and scaling them to the same size
    anom_maps_sized_resized_agg_res_lvls = []
   # Set the resolution level to be processed
    res_lvls = ([7,14])
   # Traverse each resolution level
    for res_lvl in res_lvls:
        # Set the folder path based on the current resolution level
        lvl_version = args.version.replace("#", "%d"%(res_lvl))
        res_fold = os.path.join(fold_path, lvl_version + "/rep_num_%d"%(rep_num), mvtype)

        anom_maps = []
        anom_lens = []
        # Gets the list of files in the current path and sorts them
        res_folds = os.listdir(res_fold)
        res_folds.sort()
        # Traverse the files in the folder
        for j, file_name in enumerate(res_folds):
            file_path = os.path.join(res_fold, file_name)
            # If the file name contains "anom_maps", the array is loaded and processed
            if ("anom_maps" in file_name):

                            i = int(file_name.split("_")[0])

                            array = np.load(file_path)
                            array = array[set_string]

                            anom_maps.append(array)
                            anom_lens.append(i)

            else:
                    gt_labels = np.load(file_path)
        # Calculates the maximum length and initializes the resized exception mapping array
        max_length = int(np.sqrt(np.max(anom_lens)))#If an error occurs, you can try to change the value of max_length to 4, but the fixed value will not cause unknown problems
        resized_anom_maps = np.zeros((anom_maps[0].shape[1],  len(anom_maps), max_length, max_length)) # samplex X super patch size X len X len

        anom_maps_sized_resized_agg_super_pixels = []
        # Resize and store in the resized exception map array
        for i, length in enumerate(anom_lens):

            anom_maps_sized = np.transpose(anom_maps[i], (1,0))

            length_i = int(np.sqrt(np.max(length)))

            for img_ind in range(anom_maps[0].shape[1]):
                squared_map = np.reshape(anom_maps_sized[img_ind],(length_i,length_i))
                squared_map_resized = cv2.resize(squared_map, dsize=(max_length, max_length), interpolation=cv2.INTER_CUBIC)
                resized_anom_maps[img_ind, i] = squared_map_resized
        # Adds the resized exception mapping array to the result list
        resized_anom_maps_res_lvls.append(np.stack(resized_anom_maps))
        anom_maps_sized_resized_agg_res_lvls.append(anom_maps_sized_resized_agg_super_pixels)
   # Converts the result list to a NumPy array and performs a dimension transpose
    resized_anom_maps_res_lvls = np.stack(resized_anom_maps_res_lvls) # res_lvls X samplex X super patch size X len X len
    resized_anom_maps_res_lvls = np.transpose(resized_anom_maps_res_lvls, (1, 0, 2, 3, 4)) # samplex X res_lvls X super patch size X len X len
   # The array is average pooled to reduce dimensions
    resized_anom_maps_res_lvls = np.mean(resized_anom_maps_res_lvls, 3)
    resized_anom_maps_res_lvls = np.mean(resized_anom_maps_res_lvls, 3)

    return resized_anom_maps_res_lvls, gt_labels

for mvtype in mvtype_list:
    # Initializes two empty lists to store exception images for the test set and the validation set
    resized_anom_maps_res_lvls = []
    resized_anom_maps_res_lvls_valid = []
    # Gets the anomaly image of the test set and the corresponding label
    resized_anom_maps_res_lvls, gt_labels = get_anom_map(resized_anom_maps_res_lvls, set_string = "anom_map_test")
    # Gets the exception image of the verification set and the corresponding label
    resized_anom_maps_res_lvls_valid, _ = get_anom_map(resized_anom_maps_res_lvls_valid, set_string = "anom_map_valid")
    # Scale the exception images for the test set and validation set to 224x224 size
    resized_anom_maps_res_lvls_224 = get_anom_map_224(resized_anom_maps_res_lvls, set_string = "anom_map_test")
    resized_anom_maps_res_lvls_valid_224 = get_anom_map_224(resized_anom_maps_res_lvls_valid, set_string = "anom_map_valid")
    # Traverse each pixel of the abnormal image
    for i in range(resized_anom_maps_res_lvls.shape[1]):
        for j in range(resized_anom_maps_res_lvls.shape[2]):
            # The abnormal images of the test set are normalized
            norm_fac = np.mean(resized_anom_maps_res_lvls_valid[:,i,j]) + 0.00001
            resized_anom_maps_res_lvls[:,i,j] = resized_anom_maps_res_lvls[:,i,j]/norm_fac + 0.00001
            # Calculates the AUC value of the current pixel point
            roc = roc_auc_score(gt_labels, resized_anom_maps_res_lvls[:,i,j])
    # Further processing of test set exception images scaled to 224x224
    for i in range(resized_anom_maps_res_lvls_224.shape[0]):
        # The abnormal images of the test set are normalized
        norm_fac = np.mean(resized_anom_maps_res_lvls_valid_224[i]) + reg_eps
        resized_anom_maps_res_lvls_224[i] = resized_anom_maps_res_lvls_224[i]/norm_fac + reg_eps
        # Calculates the AUC value of the current image
        roc = roc_auc_score(gt_labels, resized_anom_maps_res_lvls_224[i])
    # Calculates the average image of all test set exception images scaled to 224x224
    resized_anom_maps_res_lvls_224_mean = np.median(resized_anom_maps_res_lvls_224, 0)
    # Calculate the AUC value of the average image
    roc = roc_auc_score(gt_labels, resized_anom_maps_res_lvls_224_mean)
    # Save the original anomaly image for subsequent calculation
    resized_anom_maps_res_lvls_pre_mean = resized_anom_maps_res_lvls
    # Calculate the average value of test set anomaly images in time and space
    resized_anom_maps_res_lvls_mean = np.mean(resized_anom_maps_res_lvls, axis = 2)
    resized_anom_maps_res_lvls_mean = np.mean(resized_anom_maps_res_lvls_mean, axis = 1)
    # The final AUC value is calculated by combining the anomaly images averaged over time and space and the average images scaled to 224x224
    roc = roc_auc_score(gt_labels, resized_anom_maps_res_lvls_mean + resized_anom_maps_res_lvls_224_mean*args.lambda_factor)
    # Adds the AUC value of the current mvtype to the list
    mvtypes_roc.append(roc)
# Prints the AUC value for each mvtype and the average AUC value
for i, roc in enumerate(mvtypes_roc):
    print("Accuracy on %s is %.2f" % (mvtype_list[i],roc))

print("mean_roc %.3f"%np.mean(mvtypes_roc))

