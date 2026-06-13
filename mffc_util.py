import librosa
import numpy as np
import scipy
from matplotlib import pyplot as plt

from util import read_signal_from_wav, plot_signal

def analyse_signals(file_name_one, file_name_two, low_freq=300, high_freq=8000, n_mfcc=12, nfft=1024, frame_time=0.020, frame_step_percentage=0.5):
    signals = []
    sampling_rates = []
    mfccs = []
    deltas = []
    delta_deltas = []

    for file_name in [file_name_one, file_name_two]:
        signal, sampling_rate = read_signal_from_wav(file_name)
        signals.append(signal)
        sampling_rates.append(sampling_rate)

        frame_step = int((frame_time * frame_step_percentage) * sampling_rate)
        frame_length = int(round(int(frame_time * sampling_rate)))

        mfcc = librosa.feature.mfcc(
            y=np.asarray(signal, dtype=np.float32),
            sr=sampling_rate,
            n_mfcc=n_mfcc,
            dct_type=2,
            norm='ortho',
            n_fft=nfft,
            hop_length=frame_step,
            win_length=frame_length,
            window=scipy.signal.windows.hann,
            n_mels=26,
            fmin=low_freq,
            fmax=high_freq
        )
        mfccs.append(mfcc)
        deltas.append(librosa.feature.delta(mfcc))
        delta_deltas.append(librosa.feature.delta(mfcc, order=2))

    labels = [file_name_one, file_name_two]
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("Signal Analysis Comparison", fontsize=16, fontweight='bold')

    for col in range(2):
        sr = sampling_rates[col]
        signal = signals[col]
        times = np.arange(len(signal)) / sr

        # Row 0: Raw signal waveform
        axes[0, col].plot(times, signal, linewidth=0.5)
        axes[0, col].set_title(f"Waveform\n{labels[col]}")
        axes[0, col].set_xlabel("Time (s)")
        axes[0, col].set_ylabel("Amplitude")

        # Row 1: MFCC
        img1 = librosa.display.specshow(mfccs[col], sr=sr, fmin=low_freq, fmax=high_freq,
                                         x_axis='time', ax=axes[1, col])
        axes[1, col].set_title(f"MFCC\n{labels[col]}")
        fig.colorbar(img1, ax=axes[1, col])

        # Row 2: Delta
        img2 = librosa.display.specshow(deltas[col], sr=sr, x_axis='time', ax=axes[2, col])
        axes[2, col].set_title(f"Delta MFCC\n{labels[col]}")
        fig.colorbar(img2, ax=axes[2, col])

        # Row 3: Delta-Delta
        img3 = librosa.display.specshow(delta_deltas[col], sr=sr, x_axis='time', ax=axes[3, col])
        axes[3, col].set_title(f"Delta-Delta MFCC\n{labels[col]}")
        fig.colorbar(img3, ax=axes[3, col])

    plt.tight_layout()
    plt.show()


def analysis(file_name, low_freq=300, high_freq=8000, n_mfcc=12, nfft=1024, frame_time=0.020, frame_step_percentage=0.5):
    signal, sampling_rate = read_signal_from_wav(file_name)

    plot_signal(signal, sampling_rate)

    frame_step = int((frame_time * frame_step_percentage) * sampling_rate)  # 50 % od 20 ms -> 10 ms
    frame_length = int(round(int(frame_time * sampling_rate)))  # 20 ms

    mfcc = librosa.feature.mfcc(y=np.asarray(signal, dtype=np.float32), sr=sampling_rate, n_mfcc=n_mfcc, dct_type=2,
                                norm='ortho', n_fft=nfft, hop_length=frame_step, win_length=frame_length,
                                window=scipy.signal.windows.hann, n_mels=26, fmin=low_freq, fmax=high_freq)

    librosa.display.specshow(mfcc, sr=sampling_rate, fmin=low_freq, fmax=high_freq)

    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    plt.figure()
    librosa.display.specshow(delta, x_axis='time')
    plt.colorbar()
    plt.show()

    plt.figure()
    librosa.display.specshow(delta_delta, x_axis='time')
    plt.colorbar()
    plt.show()
