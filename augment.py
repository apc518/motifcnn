import random
from math import ceil

import numpy as np
import soundfile as sf
import pyrubberband
from pydub import AudioSegment, effects
from pydub.generators import WhiteNoise


### Distortion

def distort(clip : AudioSegment, outpath_stub, duplicate_original=True):
    """
    ## Distort clip at various levels and output to mp3 files.

    ### Arguments:
    - `outpath_stub` -- destination path including name of original file.\ 
    Ex: with `outpath_stub='./distorted_clips/myclip1.mp3'` the actual outputs might look like:\ 
    './distorted_clips/myclip1.mp3_copy.mp3'\ 
    './distorted_clips/myclip1.mp3_distorted_0.mp3'\ 
    './distorted_clips/myclip1.mp3_distorted_1.mp3'

    - `duplicate_original` -- whether or not to include a copy of the original in the output\ 
    defaults to `True`
    """

    clip = effects.normalize(clip)
    
    if duplicate_original:
        clip.export(f"{outpath_stub}_copy.mp3", format="mp3")
    
    for x in range(2):
        clip = clip + 8
        clip.export(f"{outpath_stub}_distorted_{x}.mp3", format="mp3")


### Noise

def addnoise(clip : AudioSegment, outpath_stub, duplicate_original=True):
    """
    ## Combine clip with varying levels of white noise and output to mp3 files.

    ### Arguments:
    - `outpath_stub` -- destination path including name of original file.\ 
    Ex: with `outpath_stub='./noisy_clips/myclip1.mp3'` the actual outputs might look like:\ 
    './noisy_clips/myclip1.mp3_copy.mp3'\ 
    './noisy_clips/myclip1.mp3_noisy_0.mp3'\ 
    './noisy_clips/myclip1.mp3_noisy_1.mp3'

    - `duplicate_original` -- whether or not to include a copy of the original in the output\ 
    defaults to `True`
    """

    clip = effects.normalize(clip)

    if(duplicate_original):
        clip.export(f"{outpath_stub}_copy.mp3", format="mp3")

    noise = WhiteNoise().to_audio_segment(duration=len(clip))

    noise_levels = [random.uniform(-40, -30), random.uniform(-30, -20)]
    for i in range(0, len(noise_levels)):
        combined = noise.overlay(clip, gain_during_overlay=noise_levels[i])
        combined.export(f"{outpath_stub}_noisy_{i}.mp3", format="mp3")




#######################
### Time Stretching ###
#######################

# this was taken from https://stackoverflow.com/questions/58810035/converting-audio-files-between-pydub-and-librosa also using https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples
def audiosegment_to_numpy_array(audiosegment):
    """ extracts and returns the numpy array of samples from a pydub AudioSegment """

    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)

    return fp_arr

def make_chunks(audio_segment, chunk_length):
    """
    Breaks an AudioSegment into chunks that are <chunk_length> milliseconds
    long.
    i.e. if chunk_length is 500 then you'll get a list of 500ms long audio
    segments back (except the last one, which can be shorter)
    """
    number_of_chunks = ceil(len(audio_segment) / float(chunk_length))
    return [audio_segment[i * chunk_length:(i + 1) * chunk_length] for i in range(number_of_chunks)]

def timestretch(clip : AudioSegment, outpath_stub):
    """## Timestretch segments of clip at varying speeds and output to wav files.

    ### Arguments:
    - `outpath_stub` -- destination path including name of original file.\ 
    Ex: with `outpath_stub='./stretched_clips/myclip1.mp3'` the actual outputs might look like:\ 
    './stretched_clips/myclip1.mp3_copy.wav'\ 
    './stretched_clips/myclip1.mp3_stretched_0.wav'\ 
    './stretched_clips/myclip1.mp3_stretched_1.wav'

    - `duplicate_original` -- whether or not to include a copy of the original in the output\ 
    defaults to `True`
    """

    clip = effects.normalize(clip)
    clip.export(f"{outpath_stub}_copy.wav", format="wav")

    sr = clip.frame_rate
    clipTime = clip.duration_seconds
    equal_Points = ceil(1000 * clipTime / 5)
    # print(clipTime)
    chunks = make_chunks(clip, equal_Points)
    for x in range(2):
        combined = np.array([])
        for chunk in chunks:
            rate = random.uniform(0.7, 1.3)
            converted_to_lib = audiosegment_to_numpy_array(chunk)
            to_append = pyrubberband.pyrb.time_stretch(converted_to_lib, sr, rate)
            combined = np.append(combined, to_append)
        # print(combined)
        sf.write(f"{outpath_stub}_stretched_{x}.wav", combined, sr)

# from augment import distort, addnoise, timestretch
# from pydub import AudioSegment, effects


if __name__ == "__main__":
    import os

    for pos_or_neg in ["positive", "negative"]:    
        for index, augmentation in enumerate([timestretch, distort, addnoise]):
            indir = f"./data/solo/{pos_or_neg}" if index == 0 else f"./data/augmented_stage_{index}/{pos_or_neg}"
            outdir = f"./data/augmented_stage_{index+1}/{pos_or_neg}"
            if not os.path.isdir(f"./data/augmented_stage_{index+1}"):
                os.mkdir(f"./data/augmented_stage_{index+1}")
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            
            counter = 0
            for item in os.listdir(indir):
                # normalize volume of clip first
                clip = effects.normalize(AudioSegment.from_file(f"{indir}/{item}"))

                # apply augmentation
                print(f"Augmenting {pos_or_neg} example {item} with {augmentation.__name__}")
                augmentation(clip, f"{outdir}/{item}")

    
                # limit output for testing
                # counter += 1
                # if counter >= 3 and index == 0:
                    # break