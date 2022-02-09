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
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pydub import AudioSegment, effects

from augment import audiosegment_to_numpy_array

def convert_audio_to_spectogram(filepath):
    """
    ## create a spectrogram from an audio file
    
    #### Arguments:
    - filepath -- path to the audio file of which a spectrogram will be made
    
    #### Returns
    - A PIL.Image object of the spectrogram
    """

    # load sound with pydub
    snd = AudioSegment.from_file(filepath)
    snd = effects.normalize(snd)

    # sr == sample rate 
    sr = snd.frame_rate
    audio_data = audiosegment_to_numpy_array(snd)

    # pitch shift by random amount for additional augmentation
    audio_data = librosa.effects.pitch_shift(audio_data, sr, random.randint(-6, 12))

    vertical_res = 4096
    
    # stft is short time fourier transform
    sgram = librosa.stft(audio_data, center=False, n_fft=vertical_res, win_length=vertical_res)
    
    # convert the slices to amplitude
    sgram_db = librosa.amplitude_to_db(abs(sgram))

    _, ax = plt.subplots(figsize=(5, 5))

    librosa.display.specshow(sgram_db, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='gray')

    # plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    plt.close() # dont display the spectrogram on screen, and dont leak memory

    # crop the image to get rid of useless high and low frequencies
    im = Image.open(img_buf)
    im_cropped = im.crop((0, 150, 500, 450))
    im_resized = im_cropped.resize((100, 300))
    return im_resized


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
        Assume there is a model in the directory named model.tf
    """

    # import warnings
    # warnings.simplefilter("ignore")


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
    convert_audio_to_spectogram(audio_path).save("./spec.png")

    # preprocess the spectrogram
    image = tf.keras.preprocessing.image.load_img("./spec.png", target_size=(100, 300))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    # TODO: Andy Check Model and ensure
    # TODO: add prediction confidence
    # TODO: verify that yes is 1 and no is 0
    predictions = model.predict(input_arr)
    print(f'{predictions=}')
    # print(f'{model.classnames}')

    if np.argmax(predictions[0]) == 1:
        print(f'Yes with confidence {predictions[0][1]}')
    elif np.argmax(predictions) == 0:
        print(f'No with confidence {predictions[0][0]}')
    else:
        print("an error has occurred")

    # clean up
    os.remove("./spec.png")
