# System imports
import os

# Local imports
from config import definitions

####################################################################################################

# Useful directory paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(definitions.__file__))))
DATASET_DIR = os.path.join(ROOT_DIR, "data")
SRC_DIR = os.path.join(ROOT_DIR, "src")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

BATCH_SIZE = 1000
# series length
NA = 2
NB = 2
