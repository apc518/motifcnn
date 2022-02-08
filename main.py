import os
import tensorflow as tf
from tensorflow import keras
import librosa
from PIL import Image
import librosa.display
import matplotlib.pyplot as plt
import sys
import numpy as np


def convert_audio_to_spectogram(filename, outfilename=None, overwrite=False):
    """
    convert_audio_to_spectogram -- using librosa to plot and save a spectrogram

    Arguments:
    filename -- filepath to the file that you want to see the waveplot for
    outfilename -- filepath to the output spectrogram (must be something matplotlib.pyplot.savefig() can handle)
    overwrite -- whether to overwrite if a file already exists with the given outfilename

    Returns -- None
    """

    # sr == sampling rate
    audio_data, sr = librosa.load(filename, sr=44100)

    vertical_res = 4096

    # stft is short time fourier transform
    sgram = librosa.stft(audio_data, center=False, n_fft=vertical_res, win_length=vertical_res)

    # convert the slices to amplitude
    sgram_db = librosa.amplitude_to_db(abs(sgram))

    _, ax = plt.subplots(figsize=(5, 5))

    librosa.display.specshow(sgram_db, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='gray')

    # plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
    plt.margins(0, 0)

    if outfilename in os.listdir() and not overwrite:
        print("Given filename already exists, to overwrite pass in argument `overwrite=True`")
        return
    if outfilename is not None:
        plt.savefig(outfilename)

        # crop the image to get rid of useless high and low frequencies
        im = Image.open(outfilename)
        im_cropped = im.crop((0, 90, 360, 315))
        im_cropped.save(outfilename)

    plt.close()  # dont display the spectrogram on screen


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
        Assume there is a model in the directory named model.tf
    """

    # load model
    model_path = "./model.tf"
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
    else:
        print("Model not found in directory\n")
        model_abs_path = input("Please input an absolute path")
        while os.path.exists(model_abs_path):
            print("Model not found in directory\n")
            model_abs_path = input("Please input an absolute path")
    # Verify audio
    audio_path = ""

    if len(sys.argv) == 2 and os.path.exists(sys.argv[1]):
        audio_path = sys.argv[1]
    else:
        while os.path.exists(audio_path):
            audio_path = input("[Usage] filepath: ")

    print("Have the audio loaded")
    # create spectrogram
    convert_audio_to_spectogram("./" + audio_path, "./spec.png")

    # preprocess the spectrogram
    image = tf.keras.preprocessing.image.load_img("./spec.png", target_size=(360, 225))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    # TODO: Andy Check Model and ensure
    # TODO: add prediction confidence
    # TODO: verify that yes is 1 and no is 0
    predictions = model.predict(input_arr)
    print(f'Prediction returns: {predictions}')
    if np.argmax(predictions[0]) == 1:
        print(f'Yes with confidence {predictions[0][1]}')
    elif np.argmax(predictions) == 0:
        print(f'No with confidence {predictions[0][0]}')
    else:
        print("an Error has occurred")


    # clean up and exit
    os.remove("./spec.png")
    exit(0)


