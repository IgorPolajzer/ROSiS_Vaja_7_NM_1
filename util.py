from matplotlib import pyplot as plt
from pydub import AudioSegment
import numpy as np


def read_signal_from_wav(input_file):
    sound = AudioSegment.from_wav(input_file)
    samples = np.frombuffer(sound.raw_data, dtype=np.int16).astype(np.float32) / 32767
    return samples, sound.frame_rate


def plot_signal(signal, sampling_rate):
    duration = len(signal) / sampling_rate

    t = np.linspace(0, 50, int(sampling_rate * duration), endpoint=False)

    plt.plot(t, signal)
    plt.xlabel('Čas [s]')
    plt.ylabel('Amplituda')
    plt.grid(True)
    plt.show()