import sys
import math
import numpy as np
from random import randint, seed
import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# Addition layer
#############################################
def add(input_tensor, other_tensor):
    return torch.add(input_tensor, other_tensor)

#############################################
# 2D Convolution
#############################################
def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, nonlinearity="relu"):
    layers = []

    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

    if (nonlinearity=="relu"):
        layers.append(nn.ReLU())
    elif (nonlinearity=="leaky_relu"):
        layers.append(nn.LeakyReLU())
    elif (nonlinearity=="sigmoid"):
        layers.append(nn.Sigmoid())
    elif (nonlinearity=="tanh"):
        layers.append(nn.Tanh())
    elif(nonlinearity!="none"):
        print("ERROR: Unkown nonlinearity function: '{}'".format(nonlinearity))
        sys.exit(-1)

    return nn.Sequential(*layers)

#############################################
# 2D Convolution with Batch Normalization
#############################################
def conv2d_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, nonlinearity="relu"):
    layers = []

    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    layers.append(nn.BatchNorm2d(out_channels))

    if (nonlinearity=="relu"):
        layers.append(nn.ReLU())
    elif (nonlinearity=="leaky_relu"):
        layers.append(nn.LeakyReLU())
    elif (nonlinearity=="sigmoid"):
        layers.append(nn.Sigmoid())
    elif (nonlinearity=="tanh"):
        layers.append(nn.Tanh())
    elif(nonlinearity!="none"):
        print("ERROR: Unkown nonlinearity function: '{}'".format(nonlinearity))
        sys.exit(-1)

    return nn.Sequential(*layers)

#############################################
# 2D Deconvolution
#############################################
def deconv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, bias=True, nonlinearity="relu"):
    layers = []

    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))

    if (nonlinearity=="relu"):
        layers.append(nn.ReLU())
    elif (nonlinearity=="leaky_relu"):
        layers.append(nn.LeakyReLU())
    elif (nonlinearity=="sigmoid"):
        layers.append(nn.Sigmoid())
    elif (nonlinearity=="tanh"):
        layers.append(nn.Tanh())
    elif(nonlinearity!="none"):
        print("ERROR: Unkown nonlinearity function: '{}'".format(nonlinearity))
        sys.exit(-1)

    return nn.Sequential(*layers)

#############################################
# 2D Deconvolution with Batch Normalization
#############################################
def deconv2d_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, bias=False, nonlinearity="relu"):
    layers = []

    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))
    layers.append(nn.BatchNorm2d(out_channels))

    if (nonlinearity=="relu"):
        layers.append(nn.ReLU())
    elif (nonlinearity=="leaky_relu"):
        layers.append(nn.LeakyReLU())
    elif (nonlinearity=="sigmoid"):
        layers.append(nn.Sigmoid())
    elif (nonlinearity=="tanh"):
        layers.append(nn.Tanh())
    elif(nonlinearity!="none"):
        print("ERROR: Unkown nonlinearity function: '{}'".format(nonlinearity))
        sys.exit(-1)

    return nn.Sequential(*layers)

#############################################
# 3D Convolution
#############################################
def conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, nonlinearity="relu"):
    layers = []

    layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))

    if (nonlinearity=="relu"):
        layers.append(nn.ReLU())
    elif (nonlinearity=="leaky_relu"):
        layers.append(nn.LeakyReLU())
    elif (nonlinearity=="sigmoid"):
        layers.append(nn.Sigmoid())
    elif (nonlinearity=="tanh"):
        layers.append(nn.Tanh())
    elif(nonlinearity!="none"):
        print("ERROR: Unkown nonlinearity function: '{}'".format(nonlinearity))
        sys.exit(-1)

    return nn.Sequential(*layers)

#############################################
# 3D Convolution with Batch Normalization
#############################################
def conv3d_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, nonlinearity="relu"):
    layers = []

    layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    layers.append(nn.BatchNorm3d(out_channels))

    if (nonlinearity=="relu"):
        layers.append(nn.ReLU())
    elif (nonlinearity=="leaky_relu"):
        layers.append(nn.LeakyReLU())
    elif (nonlinearity=="sigmoid"):
        layers.append(nn.Sigmoid())
    elif (nonlinearity=="tanh"):
        layers.append(nn.Tanh())
    elif(nonlinearity!="none"):
        print("ERROR: Unkown nonlinearity function: '{}'".format(nonlinearity))
        sys.exit(-1)

    return nn.Sequential(*layers)

#############################################
# 3D Deconvolution with Batch Normalization
#############################################
def deconv3d_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, bias=False, nonlinearity="relu"):
    layers = []

    layers.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias))
    layers.append(nn.BatchNorm3d(out_channels))

    if (nonlinearity=="relu"):
        layers.append(nn.ReLU())
    elif (nonlinearity=="leaky_relu"):
        layers.append(nn.LeakyReLU())
    elif (nonlinearity=="sigmoid"):
        layers.append(nn.Sigmoid())
    elif (nonlinearity=="tanh"):
        layers.append(nn.Tanh())
    elif(nonlinearity!="none"):
        print("ERROR: Unkown nonlinearity function: '{}'".format(nonlinearity))
        sys.exit(-1)

    return nn.Sequential(*layers)

