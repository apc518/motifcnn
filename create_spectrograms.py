"""
Usage: `python create_spectrogram.py <dir>`

Takes in command line argument for directory containing the data in the format:

data
├── negative
└── positive

The parent directory doesn't have to be named 'data' but the children directories must be called "negative" and "positive". 
Inside of "positive" and "negative" should be only audio files
"""

import os
import sys
import random
from multiprocessing import Process, Manager
import io

from PIL import Image
import librosa
import librosa.display
import librosa.effects
import matplotlib.pyplot as plt
import matplotlib
from pydub import AudioSegment, effects

from augment import audiosegment_to_numpy_array

matplotlib.use("Agg")

def convert_audio_to_spectogram(filepath, normalize=False, augment=True):
    """
    ## create a spectrogram from an audio file
    
    #### Arguments:
    - filepath -- path to the audio file of which a spectrogram will be made
    - normalize -- whether to normalize audio before converting it to a spectrogram
    - augment -- whether to apply augmentation (pitch shifting) before turning audio into spectrogram
    
    #### Returns
    - A PIL.Image object of the spectrogram
    """
    
    # load sound with pydub
    snd = AudioSegment.from_file(filepath)
    if normalize:
        snd = effects.normalize(snd)

    # sr == sample rate 
    sr = snd.frame_rate
    audio_data = audiosegment_to_numpy_array(snd)

    # pitch shift by random amount for additional augmentation
    if augment:
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

def save_spectrograms_as_png(specs):
    for img, outpath in specs:
        img.save(outpath)

def convert_specs(inputs, return_list):
    # all files in the inputs list to spectrograms
    images = []
    counter = 0
    for item in inputs:
        parent_dir = item.rsplit("/", 2)[1]
        filename_base = item.rsplit("/", 1)[1].rsplit(".", 1)[0]
        if (parent_dir not in ["positive", "negative"]):
            raise Exception("Parent directory of file must be either 'positive' or 'negative'.")

        img = convert_audio_to_spectogram(item)
        images.append((img, f"./data/spec/{parent_dir}/{filename_base}.png"))
        counter += 1
        print(f"{parent_dir} spectrogram {counter}/{len(inputs)}")
    
    return_list.append(images)

# from https://stackoverflow.com/a/2135920
def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

if __name__ == "__main__":
    data_dir = sys.argv[1]
    if data_dir.endswith("/"): # trim trailing slash
        data_dir = data_dir[:-1]

    # ensure output directories are present
    if not os.path.isdir("./data/spec/positive"):
        os.makedirs("./data/spec/positive")

    if not os.path.isdir("./data/spec/negative"):
        os.makedirs("./data/spec/negative")

    # full list of all target files
    target_files = [f'{data_dir}/positive/{x}' for x in os.listdir(f'{data_dir}/positive')]
    target_files += [f'{data_dir}/negative/{x}' for x in os.listdir(f'{data_dir}/negative')]

    random.shuffle(target_files) # if this script is stopped or fails part of the way through, we want the output to be evenly mixed

    # process in batches so that we don't run out of memory and give the CPU a few seconds to cool off every now and then
    batches = split(target_files, round(len(target_files) / 1000))

    print(f"Processing data with {len(batches)} batches of size ~{len(batches[0])} each.")

    num_processes = 6
    for i, batch in enumerate(batches):
        print(f"beginning batch {i+1}")

        workloads = split(batch, num_processes)

        print("Length of each workload in this batch:", [len(x) for x in workloads])

        manager = Manager()
        return_list = manager.list()

        jobs = []
        for sublist in workloads:
            p = Process(target=convert_specs, args=(sublist, return_list))
            jobs.append(p)
            p.start()
        
        for p in jobs:
            p.join()
        
        for lst in return_list:
            save_spectrograms_as_png(lst)