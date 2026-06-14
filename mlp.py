import os
import h5py
import keras
import librosa
import numpy as np
import tensorflow as tf
from keras import models, layers
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt

from scipy import signal
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def extract_features(y, sr, n_mfcc_coef, n_fft=512, hop_length=160, window=signal.windows.hamming(512), fmin=300, fmax=8000, verbose=False):
    vectors = []

    # 1. Binary mask (True/False) representing if the signal crossed the x axis or not.
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=int(sr/4))
    vectors.append(np.mean(zcr).reshape(1))

    # 2. Estimated tempo (BPM) vector (multi-chanel signal -> 1 tempo per channel).
    tempo = librosa.feature.tempo(y=y, sr=sr)
    vectors.append(tempo)

    # 3. 4. 5. MFCC, Delta, Delta-delta
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc_coef,
                                n_fft=n_fft, hop_length=hop_length,
                                window=window, fmin=fmin, fmax=fmax,
                                n_mels=128)
    delta = librosa.feature.delta(mfcc)
    delta_delta = librosa.feature.delta(mfcc, order=2)

    for i in range(0, n_mfcc_coef):
        vectors.append(np.mean(mfcc[i]))
        vectors.append(np.std(mfcc[i]))
        vectors.append(np.mean(delta[i]))
        vectors.append(np.std(delta[i]))
        vectors.append(np.mean(delta_delta[i]))
        vectors.append(np.std(delta_delta[i]))

    # 6. Chroma STFT (harmonic/pitch content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft)  # shape: (12, n_frames) - 12 pitch classes
    vectors.append(np.mean(chroma))
    vectors.append(np.std(chroma))

    # 7. Tonnetz (tonal relationships)
    y_harmonic = librosa.effects.harmonic(y)  # Harmonic component
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    vectors.append(np.mean(tonnetz))
    vectors.append(np.std(tonnetz))

    # 8. Spectral contrast (difference betwee peaks and valleys in spectrum of frequency bands.)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft)
    for i in range(contrast.shape[0]):
        vectors.append(np.mean(contrast[i]))
        vectors.append(np.std(contrast[i]))

    # 9. RMS energija (root mean square value of a waveform.)
    rms = librosa.feature.rms(y=y)
    vectors.append(np.mean(rms))
    vectors.append(np.std(rms))

    flat = np.concatenate([x.ravel() for x in vectors])

    if verbose:
        print(f"Tempo: {tempo.shape}, ZCR: {zcr.shape}, MFCC: {mfcc.shape}, Delta: {delta.shape}, "
              f"Delta-delta: {delta_delta.shape}, Chroma: {chroma.shape}, Tonnetz: {tonnetz.shape}, "
              f"Contrast: {contrast.shape}, RMS: {rms.shape}, Flat: {flat.shape}")

    return flat


def generate_dataset(dataset_name, genres, n_mfcc_coef, n_parts_sig_range):
    n_genres = len(genres)

    # Dataset - Will take some time to generate
    data = []
    data_labels = []

    for i_genre in range(0, n_genres):
        print(f"genre: {genres[i_genre]}")
        for filename in os.listdir(
                f'/home/igor/Desktop/MAG/1_LETNIK/2_SEMESTER/RACUNALNISKA_OBDELAVA_SIGNALOV_IN_SLIK/Vaja_7/genres/{genres[i_genre]}'):
            fn = f'/home/igor/Desktop/MAG/1_LETNIK/2_SEMESTER/RACUNALNISKA_OBDELAVA_SIGNALOV_IN_SLIK/Vaja_7/genres/{genres[i_genre]}/{filename}'

            # There is one problematic file - format problem (can try ffmpeg decoder) - pip install ffmpeg-python
            try:
                # Load file (sig-signal; sr-sampling rate)
                sig, sr = librosa.load(fn, mono=True, duration=28)  # Load 28 seconds of the file

                # Split signals into smaller chunks (different counts).

                for n_parts_sig in n_parts_sig_range:
                    for y in np.split(sig, n_parts_sig):
                        # Features - Data

                        ef = extract_features(y, sr, n_mfcc_coef, verbose=False)

                        if len(ef) < 82:
                            print(f"File: {genres[i_genre]}/{filename}, features: {len(ef)}")
                        else:
                            data.append(ef)

                        # Genre - Label
                        data_labels.append(i_genre)
            except Exception as e:
                print(f"ERROR {filename}: {e}")

    # Covert to numpy arrays
    data = np.array(data)
    data_labels = np.array(data_labels)

    print("Data size:", np.shape(data))
    print("Data labels size:", np.shape(data_labels))

    # Save to h5 file
    hf = h5py.File(dataset_name, 'w')
    hf.create_dataset('data', data=data)
    hf.create_dataset('data_labels', data=data_labels)
    hf.close()


def load_dataset(dataset_name):
    # Load dataset from h5 file
    hf = h5py.File(dataset_name, 'r')

    data = hf.get('data')
    data = np.array(data)

    data_labels = hf.get('data_labels')
    data_labels = np.array(data_labels)

    print('Data size:', np.shape(data))
    print('Data_labels size:', np.shape(data_labels))

    hf.close()

    return data, data_labels


def normalize_and_split_dataset(data, data_labels):
    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data, dtype=float))

    # Split into test and train
    X_train, X_test, y_train, y_test = train_test_split(X, data_labels, test_size=0.2, stratify=data_labels)

    # Split into train and valid
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

    # Sizes
    print('Train:', np.shape(y_train))
    print('Test:', np.shape(y_test))
    print('Val:', np.shape(y_val))

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_dataset_labels(y_train, y_test, y_val, genres):
    n_genres = len(genres)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, data, title in zip(axes, [y_train, y_test, y_val], ['Train', 'Test', 'Validation']):
        ax.hist(data, bins=n_genres, rwidth=0.7)
        ax.set_xlabel('Genres')
        ax.set_ylabel('Frequency')
        ax.set_title(f"{title} dataset")
        ax.set_xticks(range(n_genres))
        ax.set_xticklabels(genres, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def train_model(X_train, y_train, X_val, y_val, n_genres):
    # NN
    model = models.Sequential()
    model.add(layers.Input((X_train.shape[1],)))

    model.add(layers.Dense(256, activation='relu'))  # Prva skrita plast
    model.add(layers.Dropout(0.3))  # Preprečuje overfitting

    model.add(layers.Dense(128, activation='relu'))  # Postopno zmanjševanje
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64, activation='relu'))  # Zadnja skrita plast pred izhodom
    model.add(layers.Dropout(0.2))  # Manjši dropout

    model.add(layers.Dense(n_genres, activation='softmax'))  # Izhod - 10 žanrov

    opt = keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metr = keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer=opt, loss=loss, metrics=[metr])

    model.summary()

    # Stopping criterion to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Save best weights
    model_checkpoint = ModelCheckpoint("mlp.weights.h5", save_best_only=True, save_weights_only=True)

    # Train
    t_epochs = 150
    b_size = 32
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=t_epochs, batch_size=b_size,
                        callbacks=[early_stopping, model_checkpoint])

    # Load best weights
    model.load_weights("mlp.weights.h5")

    return model, history


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    # Lets observe the loss metric on both the training (blue) and validation (orange) set
    # What do we noice?
    axes[0].plot(history.history['loss'], label='train')
    axes[0].plot(history.history['val_loss'], label='val')
    axes[0].legend()
    axes[0].set_title('Loss')

    axes[1].plot(history.history['sparse_categorical_accuracy'], label='train')
    axes[1].plot(history.history['val_sparse_categorical_accuracy'], label='val')
    axes[1].legend()
    axes[1].set_title('Accuracy')


def evaluate_model(model, X_test, y_test, X_train, y_train, genres):
    n_genres = len(genres)
    # Now to evaluate our model on train and test data

    # Train NN
    test_loss, test_acc = model.evaluate(X_train, y_train, verbose=0)
    print('Acc train NN: %.3f' % test_acc)

    # Test NN
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Acc test NN: %.3f' % test_acc)

    # Test NN
    # Predictions for additional analysis
    predictions = model.predict(X_test)

    # Confusion matrix
    predicted_labels = np.argmax(predictions, axis=1)
    conf = confusion_matrix(y_test, predicted_labels, normalize="pred")  # Normalize pred! Explain why?

    # Visualise confusion matrix
    plt.figure()
    plt.imshow(conf)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.yticks(np.arange(n_genres), genres)
    plt.xticks(np.arange(n_genres), genres, rotation='vertical')
    plt.colorbar()
    plt.show()