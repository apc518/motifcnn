"""
Usage: `python normalize.py <data_path>`

data_path is a directory containing two directories each of which contain audio files
"""

import os
import sys

from pydub import AudioSegment, effects

def detect_leading_silence(sound : AudioSegment, silence_threshold=-18, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    sound = effects.normalize(sound)

    trim_ms = 0  # ms
    assert chunk_size > 0  # to avoid infinite loop

    while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def trimmed_silence(sound : AudioSegment):
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())
    duration = len(sound)
    return sound[start_trim:duration - (end_trim - 300)]

def trimmed_silence_from_file(filepath):
    return trimmed_silence(AudioSegment.from_file(filepath))

def normalize(data_dir):
    out_dir = f"{data_dir}_norm"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for category in os.listdir(data_dir):
        if not os.path.isdir(f"{data_dir}/{category}"):
            continue
        if not os.path.exists(f"{out_dir}/{category}"):
            os.mkdir(f"{out_dir}/{category}")
        
        counter = 0
        for item in os.listdir(f"{data_dir}/{category}"):
            print(f"trimming {item}")

            trimmed_item = trimmed_silence(AudioSegment.from_file(f"{data_dir}/{category}/{item}"))
            normalized_item = effects.normalize(trimmed_item)
            normalized_item.export(f"{out_dir}/{category}/{item}_trimmed.mp3", format='mp3')
            
            # limit for testing purposes
            # if counter >= 5:
            #     break
            # counter += 1
    
    return out_dir

if __name__ == "__main__":
    if len(sys.argv) > 1:
        normalize(sys.argv[1])
    else:
        print("Usage: `python normalize.py <data_path>`")