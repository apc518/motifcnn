"""
Usage: python main.py <pathtomodel>
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tensorflow warning messages about not having a cuda gpu
import sys

import tensorflow as tf
from tensorflow import keras
import numpy as np

from create_spectrograms import convert_audio_to_spectogram


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
        Assume there is a model in the directory named model.tf
    """

    # import warnings
    # warnings.simplefilter("ignore")

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
        
    while True:
        # get audio file from user
        audio_path = input("Classify audio file: ")
        if audio_path in ["exit", "stop", "quit"]:
            break
        if audio_path == "":
            continue

        try:
            # create spectrogram
            spec = convert_audio_to_spectogram(audio_path, normalize=True, augment=False)
            spec.save(spec_temp_file_name)

            # preprocess the spectrogram
            image = tf.keras.preprocessing.image.load_img(spec_temp_file_name, target_size=(100, 300))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])

            predictions = model.predict(input_arr)

            if np.argmax(predictions[0]) == 1:
                print(f'Yes ({(100 * predictions[0][1]):.2f}%)')
            elif np.argmax(predictions[0]) == 0:
                print(f'No ({(100 * predictions[0][0]):.2f}%)')
            else:
                print("an error has occurred")
            
            # print(f'\t{predictions=}')

            # clean up
            os.remove(spec_temp_file_name)
        except Exception as e:
            print(e)
