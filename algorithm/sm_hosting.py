
import optparse
import os
import json
import pickle
from io import StringIO

import pandas as pd

# TODO: clean all this up once the paths settle down
# TODO: Preprocessing code for MSG-net
# TODO:


# Magic folders :)
prefix = '/opt/ml/'

# Input data location
input_path = prefix + 'input/data'
# Output data: success/error, data.
output_path = os.path.join(prefix, 'output')
# Model location
model_path = os.path.join(prefix, 'model')
# Hyperparameters
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

# Model file name
model_file = os.path.join(model_path,'neural_style_transfer.model')

# Styles

style_folder = ''

style_size = 512

cuda = 0

# ngf
ngf = 128


