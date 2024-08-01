import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}



@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
def main(**kwargs):
    pass


'''
A function called run is defined, which takes multiple arguments:
    methods: A dictionary containing various methods, such as data loader, sampler acquisition method, PatchCore model acquisition method, etc.
    results_path: indicates the path to store the results.
    gpu: GPU device to use.
    seed: Random number seed, used to ensure repeatability of the experiment.
    log_group: indicates the log group, which is used to organize logs.
    log_project: Log item, used to identify the log source.
    save_segmentation_images: A Boolean value indicating whether to save the split image.
    save_patchcore_model: A Boolean value indicating whether to save the PatchCore model.
'''
@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
):
    #Unpack the methods parameters into a dictionary
    methods = {key: item for (key, item) in methods}
    # Create a folder to store the results
    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )
    # Gets the data loader using the given seed
    list_of_dataloaders = methods["get_dataloaders"](seed)
    # Set up the Torch device, depending on whether the GPU is available
    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    # The device context is used to handle GPU-specific operations because there are GPU memory leaks that need to be addressed using the context manager
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )
    # Initializes an empty list to collect results
    result_collect = []
    # Traverse each collection of data loaders
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        # Record the data set name and progress of the current evaluation
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )
        # Set up seeds to ensure repeatability of the experiment
        patchcore.utils.fix_seeds(seed, device)
        # Gets the name of the current dataset
        dataset_name = dataloaders["training"].name
        # Do the following in the device context for memory management
        with device_context:
            torch.cuda.empty_cache()
            # Gets the image size of the training dataset
            imagesize = dataloaders["training"].dataset.imagesize
            # Method of obtaining sampler
            sampler = methods["get_sampler"](
                device,
            )
            # Get a list of PatchCore models
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            # If multiple PatchCore models are used, logs are logged
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            # Train each PatchCore model
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"])

            torch.cuda.empty_cache()
            # Initializes an aggregator to collect scores and splits of predicted results
            aggregator = {"scores": [], "segmentations": []}
            # Predictions were made using each PatchCore model
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)
            # The scores predicted by the model are normalized
            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)
            # The segmentation results predicted by the model are normalized
            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)
            # Determine the label of the abnormal data based on the data set label
            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            # # (Optional) Draw a sample image
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                # Image and mask conversion function for visualization
                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                # Save split image
                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics.")
            # Calculate the image-level AUROC
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )["auroc"]

            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            # Collect the results of the current dataset
            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            # Record evaluation index
            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # # （Optional) Save the PatchCore model for later use
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and average scores in a CSV file
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
):
    '''
    backbone_names,# A list of pre-trained model names
    layers_to_extract_from,# A list of layers from which feature maps are to be extracted
    pretrain_embed_dimension,# Embeddedness of the pre-trained model
    target_embed_dimension,# Target embedding dimension
    preprocessing,# Preprocessing function
    aggregation,# Characteristic polymerization method
    patchsize,
    patchscore,
    patchoverlap,# Patch overlap size
    anomaly_scorer_num_nn,# The number of nearest neighbors used for an exception score
    patchsize_aggregate,# Aggregate patch size
    faiss_on_gpu,# Whether to use the GPU for FAISS search
    faiss_num_workers,# Number of worker threads searched by FAISS

    '''
    #Converts the backbone_names passed in to a list to ensure consistency in subsequent operations
    backbone_names = list(backbone_names)
    # If there are multiple pretrained models, handle layers_to_extract_from
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            # Process pre-trained model names with seed
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            # Load the pre-trained model
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed
            # Initializes the nearest neighbor search method
            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
            # Create a PatchCore instance
            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    # Return the name of the get_patchcore function and the function itself
    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
#This function is used to create different feature samplers based on the given name and percentage
def sampler(name, percentage):

    # Defines an internal function 'get_sampler' that takes a device argument
    def get_sampler(device):
        # Returns different feature sampler instances based on the value of the 'name' parameter
        if name == "identity":
            # If the name is 'identity', then IdentitySampler is returned, which does not make any changes to the feature
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            # If the name is 'greedy_coreset', return the GreedyCoresetSampler,
            # It uses a greedy algorithm to select a subset of features
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            # If the name is' approx_greedy_coreset, it returns ApproximateGreedyCoresetSampler,
            # It is an approximate version of the greedy algorithm, which is usually faster but may have slightly different results
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)


    return ("get_sampler", get_sampler)

@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    # Get the data set information from a global variable, which is usually a dictionary containing the name of the data set module and the class name
    dataset_info = _DATASETS[name]
    # Import the dataset module dynamically, using the __import__ function and specifying the fromlist argument
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    # Define an internal function 'get_dataloaders' that takes a random seed as an argument
    def get_dataloaders(seed):
        dataloaders = []
        # Iterate over all the subdata sets
        for subdataset in subdatasets:
            # Create an instance of the training dataset, use the classes in the dataset library, and pass in the appropriate parameters
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )
            # Create a test dataset instance, use the classes in the dataset library, and pass in the appropriate parameters
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )
            # Create a training data loader, using PyTorch's DataLoader, and pass in the appropriate parameters
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            # Create a test data loader, using PyTorch's DataLoader, and pass in the appropriate parameters
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset
            # If the split ratio between the training set and the validation set is less than 1, an instance of the validation dataset is created
            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )
                # Create a validation data loader, using PyTorch's DataLoader, and pass in the appropriate parameters
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            # Encapsulate the training, validation, and test data loader into a dictionary
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }
            # Adds the dictionary to the list of data loaders
            dataloaders.append(dataloader_dict)

        return dataloaders


    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
