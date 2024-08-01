#!/usr/bin/python
# -*- coding: utf-8 -*-
from time import time
import numpy as np
import torch.cuda
from torch import nn

def get_pdn(out=384):
    #A convolutional neural network (CNN) model is defined, which consists of multiple convolutional layers, activation functions and pooling layers
    return nn.Sequential(
        nn.Conv2d(3, 256, 4), nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(256, 512, 4), nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(512, 512, 1), nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, 3), nn.ReLU(inplace=True),
        nn.Conv2d(512, out, 4), nn.ReLU(inplace=True),
        nn.Conv2d(out, out, 1)
    )

def get_ae():
    # An Autoencoder (Autoencoder) architecture is defined, which consists of an encoder and a decoder.
    # An autoencoder is a neural network that learns to compress and decompress input data.
    # The encoder part is responsible for reducing the dimension of the input data, while the decoder attempts to reconstruct the original input data.
    return nn.Sequential(
        # encoder
        nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 8),
        # decoder
        nn.Upsample(3, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(8, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(15, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(32, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(63, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(127, mode='bilinear'),
        nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(inplace=True),
        nn.Upsample(56, mode='bilinear'),
        nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 384, 3, 1, 1)
    )

# Check whether a GPU is available
gpu = torch.cuda.is_available()

# Definition model
autoencoder = get_ae()
teacher = get_pdn(384)
student = get_pdn(768)

# Set the model to evaluation mode
autoencoder = autoencoder.eval()
teacher = teacher.eval()
student = student.eval()

#If there is a GPU, convert the model to semi-precision and move it to the GPU
if gpu:
    autoencoder.half().cuda()
    teacher.half().cuda()
    student.half().cuda()

quant_mult = torch.e    # Set the quantization multiple to e
quant_add = torch.pi    # Set quantization offset to pi

# Start testing
with torch.no_grad():
    times = []
    for rep in range(2000):
        # Create a random input image and select the data type based on whether the GPU is available
        image = torch.randn(1, 3, 256, 256, dtype=torch.float16 if gpu else torch.float32)
        start = time()# Record start time
        if gpu:
            image = image.cuda()    # If there is a GPU, move the image to the GPU

        # Use models to process images
        t = teacher(image)
        s = student(image)

        st_map = torch.mean((t - s[:, :384]) ** 2, dim=1)   # Calculate the mean square error of the first 384 channels of the teacher model and the student model
        ae = autoencoder(image) # Image processing using an autoencoder model
        ae_map = torch.mean((ae - s[:, 384:]) ** 2, dim=1)  # Calculate the mean square error of 384 channels after the autoencoder output and student model
        st_map = st_map * quant_mult + quant_add    # Quantization multiples and shifts are applied to the teacher-student mean square error
        ae_map = ae_map * quant_mult + quant_add    # Quantization multiples and shifts are applied to the autoencod-student mean square error
        result_map = st_map + ae_map    # The resulting graph is obtained by superimposing two mean-square graphs
        result_on_cpu = result_map.cpu().numpy()    # Move the resulting graph to the CPU and convert it to a NumPy array
        timed = time() - start # Calculated processing time
        times.append(timed) # Record the processing time to a list
print(np.mean(times[-1000:]))   # Output the average time of the last 1000 processing

