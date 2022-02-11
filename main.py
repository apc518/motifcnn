"""
Usage: python main.py <pathtoaudio> [<pathtomodel>]
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import sys
import io

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
    model_path = "./model.tf"
    if len(sys.argv) > 2:
        model_path = sys.argv[2]

    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        print(f"Model directory \"{model_path}\" not found.")
        exit(1)
        
    # Verify audio
    audio_path = ""

    if os.path.exists(sys.argv[1]):
        audio_path = sys.argv[1]
    else:
        print("Audio file not found.")
        exit(1)

    # create spectrogram
    spec = convert_audio_to_spectogram(audio_path, normalize=True, augment=False)
    spec.save(spec_temp_file_name)

    # preprocess the spectrogram
    image = tf.keras.preprocessing.image.load_img(spec_temp_file_name, target_size=(100, 300))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    predictions = model.predict(input_arr)
    print(f'{predictions=}')
    # print(f'{model.classnames}')

    if np.argmax(predictions[0]) == 1:
        print(f'Yes with confidence {2 * (predictions[0][1] - 0.5)}')
    elif np.argmax(predictions[0]) == 0:
        print(f'No with confidence {2 * (predictions[0][0] - 0.5)}')
    else:
        print("an error has occurred")

    # clean up
    os.remove(spec_temp_file_name)
