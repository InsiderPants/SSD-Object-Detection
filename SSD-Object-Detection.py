# -*- coding: utf-8 -*-

# Object Detection using SSD(Single Shot Detector Multibox)


#
import torch 
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio