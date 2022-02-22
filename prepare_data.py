"""
Usage: python prepare_data.py <data_path>

Takes in command line argument for directory containing the data in the format:

data
├── negative
└── positive

The parent directory doesn't have to be named 'data' but the children directories must be called "negative" and "positive". 
Inside of "positive" and "negative" should be only audio files
"""

import sys

from normalize import normalize
from augment import augment
from create_spectrograms import create_spectrograms

if len(sys.argv) > 1:
    norm_dir = normalize(sys.argv[1])
    aug_dir = augment(norm_dir)
    create_spectrograms(aug_dir)
else:
    print("Usage: python prepare_data.py <data_path>")