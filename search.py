"""
Search through all audio files recursively in a directory for The Lick, given a window size in seconds
"""

import argparse
from math import floor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tensorflow warning messages about not having a cuda gpu

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pydub import AudioSegment
from memory_profiler import profile

from create_spectrograms import audio_to_spectrogram

SPEC_TEMP_FILE = "./spec.png"

model = None
window_size = 3 # seconds
stagger = 1


def predict(model, snd):
    spec = audio_to_spectrogram(snd, normalize=True, augment=False)
    spec.save(SPEC_TEMP_FILE)

    # preprocess the spectrogram
    image = tf.keras.preprocessing.image.load_img(SPEC_TEMP_FILE, target_size=(100, 300))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    predictions = model.predict(input_arr).tolist()[0]

    # clean up
    os.remove(SPEC_TEMP_FILE)

    return predictions[0] > 0.95

# @profile
def find_lick_in_file(filepath):
    snd = AudioSegment.from_file(filepath)
    snd_len_secs = len(snd) / 1000
    for i in range(floor(snd_len_secs / window_size)):
        snd_slice = snd[i * 1000 : (i + snd_len_secs // window_size) * 1000]
        if predict(model, snd_slice):
            secs = i * window_size
            print(f"    ##### Found the lick in {filepath} at {secs // 60}:{secs % 60:02d} #####")
        else:
            print(f"\r({i}/{floor(snd_len_secs / window_size)})", end="")
        
    print()

def search_files(path):
    if os.path.isdir(path):
        for item in os.listdir(path):
            search_files(os.path.join(path, item))
    elif os.path.isfile(path):
        print(f"Searching {path}...")
        find_lick_in_file(path)

def main():
    global model, window_size, stagger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", metavar="PATH_TO_MODEL", type=str, help="path to a keras model")
    parser.add_argument("-i", "--inputpath", metavar="SEARCH_PATH", type=str, help="path to a keras model")
    parser.add_argument("-w", "--windowsize", metavar="WINDOW_SIZE", type=int, default=3, help="window size to check for The Lick, in seconds")
    parser.add_argument("-s", "--stagger", metavar="STAGGER", type=int, default=1, help="number of overlapping staggered windows")

    args = parser.parse_args()

    if args.windowsize < 1 or args.stagger < 1:
        print("Bruh negatives? tf")
        exit(0)

    if args.model is None:
        print("Need a model bruh")
        exit(0)

    if args.inputpath is None:
        print("I need a path dude :|")
        exit(0)

    
    print(f"Loading model {args.model}...")

    # load model
    model = keras.models.load_model(args.model)
    window_size = args.windowsize
    stagger = args.stagger

    search_files(args.inputpath)

    # model = keras.models.load_model("models/transfer_all_v1_15ep_0.001lr.tf")

    # search_files("lotsofsongs")
    

if __name__ == "__main__":
    main()