"""
Usage: python main.py <pathtomodel> [<test folder>]
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tensorflow warning messages about not having a cuda gpu
import sys

import tensorflow as tf
from tensorflow import keras
import numpy as np

from create_spectrograms import audio_to_spectrogram
from normalize import trimmed_silence_from_file


SPEC_TEMP_FILE = "./spec.png"


def predict(model, audio_path):
    # create spectrogram
    snd = trimmed_silence_from_file(audio_path)
    spec = audio_to_spectrogram(snd, normalize=True, augment=False)
    spec.save(SPEC_TEMP_FILE)

    # need to try/catch the rest and record if an exception occurred so that we
    # ensure the spec temp file will be removed, but also that we throw an exception to the caller

    raise_exception = True

    try:
        # preprocess the spectrogram
        image = tf.keras.preprocessing.image.load_img(SPEC_TEMP_FILE, target_size=(100, 300))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])

        predictions = model.predict(input_arr).tolist()[0]

        # some of our models have only one output neuron, others have two
        if len(predictions) == 1:
            if predictions[0] > 0.5:
                print(f'Yes ({predictions[0]:.4f})')
            else:
                print(f'No ({predictions[0]:.4f})')
        elif len(predictions) == 2:
            if predictions[1] > predictions[0]:
                print(f'Yes ({predictions[0]:.4f}, {predictions[1]:.4f})')
            else:
                print(f'No ({predictions[0]:.4f}, {predictions[1]:.4f})')
        
        raise_exception = False
    except Exception as e:
        print(e)
    
    # clean up
    os.remove(SPEC_TEMP_FILE)

    if raise_exception:
        raise Exception(f"Problem occurred on {audio_path}")



def main(model_path):
    try:
        print(f"Loading model from {model_path}...")
        model = keras.models.load_model(model_path)
        print("Model loaded. Type 'exit' to exit.")
    except Exception as e:
        if isinstance(e, OSError):
            print("Model not found.")
        else:
            print(f"Model could not be loaded.")
            print(e)
        
        exit(1)
        
    if len(sys.argv) > 2:
        test_dir = sys.argv[2]
        for category in os.listdir(test_dir):
            if not os.path.isdir(f"{test_dir}/{category}"):
                continue
            for item in os.listdir(f"{test_dir}/{category}"):
                item_path = f"{test_dir}/{category}/{item}"
                print(f"{item_path}: ", end="")
                try:
                    predict(model, f"{test_dir}/{category}/{item}")
                except:
                    break

    while True:
        # get audio file from user
        audio_path = input("Classify audio file: ")
        if audio_path in ["exit", "stop", "quit"]:
            break
        if audio_path == "":
            continue

        predict(model, audio_path)
    


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python main.py <model> [<test folder>]")