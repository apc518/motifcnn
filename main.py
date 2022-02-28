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


def predict(model, audio_path):
    try:
        # create spectrogram
        snd = trimmed_silence_from_file(audio_path)
        spec = audio_to_spectrogram(snd, normalize=True, augment=False)
        spec.save(spec_temp_file_name)

        # preprocess the spectrogram
        image = tf.keras.preprocessing.image.load_img(spec_temp_file_name, target_size=(100, 300))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])

        predictions = model.predict(input_arr).tolist()[0]

        if len(predictions) == 2:
            if predictions[1] > predictions[0]:
                print(f'Yes ({(100 * predictions[1]):.2f}%)')
            else:
                print(f'No ({(100 * predictions[0]):.2f}%)')
        elif len(predictions) == 1:
            if predictions[0] > 0.5:
                print(f'Yes ({(100 * predictions[0]):.2f}%)')
            else:
                print(f'No ({(100 * (1 - predictions[0])):.2f}%)')

        # clean up
        os.remove(spec_temp_file_name)
    except Exception as e:
        print(e)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    spec_temp_file_name = "./spec.png"

    # load model
    model_path = ""
    # if user has provided model path, use that
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    # otherwise, search for directory ending in .tf
    else: 
        for item in os.listdir():
            if item.endswith(".tf") and os.path.isdir(item):
                model_path = item
                break

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = keras.models.load_model(model_path)
        print("Model loaded. Type 'exit' to exit.")
    else:
        print(f"No model found.")
        exit(1)
        
    if len(sys.argv) > 2:
        test_dir = sys.argv[2]
        for category in os.listdir(test_dir):
            for item in os.listdir(f"{test_dir}/{category}"):
                item_path = f"{test_dir}/{category}/{item}"
                print(f"{item_path}: ", end="")
                predict(model, f"{test_dir}/{category}/{item}")

    while True:
        # get audio file from user
        audio_path = input("Classify audio file: ")
        if audio_path in ["exit", "stop", "quit"]:
            break
        if audio_path == "":
            continue

        predict(model, audio_path)
