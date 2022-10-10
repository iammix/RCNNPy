import os
import sys
import math
import random
import numpy as np
import cv2
from config import Config


class ShapeConfig(Config):
    # Give the configuration a recognizable name
    NAME = "shapes"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    NUM_CLASSES = 1 + 3  # background + 3 shapes

    IMAGE_MIN_DIM = 128

    IMAGE_MAX_DIM = 128

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    TRAIN_ROIS_PER_IMAGE = 32

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 5
