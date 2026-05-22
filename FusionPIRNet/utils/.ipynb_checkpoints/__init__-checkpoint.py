"""Useful utils
"""
from .misc import *
from .logger import *
from .eval import *

# progress bar
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
from progress.bar import Bar as Bar


from dataset.roof import Roof, Roof_crop
from model.roof_models import *
from losses.roof_loss_functions import ComputeRoofLoss
from evaluation.roof_metrics import ComputeRoofMetric
from model.roof_map_cons import RoofMapCons

__version__ = "1.0.0"
__author__ = "Roof MTL Team"

# Task definitions
ROOF_TASKS = ['seg6', 'seg9']
ROOF_CLASS_NUMBERS = {
    'seg6': 6,  # background + 5 directions  
    'seg9': 9   # background + 8 structures
}

# Class names
SEG6_CLASSES = ['background', 'N', 'E', 'S', 'W', 'flat']
SEG9_CLASSES = ['background', 'pvmodule', 'dormer', 'window', 'ladder', 'chimney', 'shadow', 'tree', 'unknown']