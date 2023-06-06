# System imports
import os

# Local imports
from config import definitions

####################################################################################################

# Useful directory paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(definitions.__file__))))
DATASET_DIR = os.path.join(ROOT_DIR, "data")
SRC_DIR = os.path.join(ROOT_DIR, "src")
MODELS_DIR = os.path.join(ROOT_DIR, "src/models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

BATCH_SIZE = 100
# series length
NA = 2
NB = 2


from enum import Enum

class ModelRepresentation(Enum):
    NARX = 0
    NOE = 1
    SS = 2

class IdentificationMethod(Enum):
    GP = 0
    NN = 1