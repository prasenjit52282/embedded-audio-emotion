import pandas as pd
import numpy as np
import librosa
import joblib

def extract_features_for_prediction(file_path):
    """Extract features for prediction from a new audio file."""
    y, sr = librosa.load(file_path, sr=None)
    
    # 1. MFCC (13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # 2. Chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # 3. Energy (1)
    energy = np.mean(librosa.feature.rms(y=y))

    # 4. Zero Crossing Rate (1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # 5. Spectral Contrast (7)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # 6. F0 (1)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    f0 = np.mean(pitches[pitches > 0])
    
    # 7. F2 (1) with shape matching
    harmonic = librosa.effects.hpss(y)[0]
    harmonic_pitches, harmonic_magnitudes = librosa.piptrack(y=harmonic, sr=sr)
    harmonic_mask = harmonic_pitches > 0  # Identify positive pitch values only

    # Ensure shapes match before applying the mask
    if harmonic_mask.shape == harmonic_pitches.shape:
        f2 = np.mean(harmonic_pitches[harmonic_mask])
    else:
        f2 = np.nan  # Assign NaN if shapes do not match to avoid indexing errors

    # 7. Jitter (1)
    jitter = np.mean(np.abs(np.diff(librosa.util.normalize(mfccs[0]))))

    # 8. Shimmer (1)
    shimmer = np.mean(np.abs(np.diff(y)))

    # 9. Band Energy Ratio (1)
    band_energy = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # 10. Pause Rate (1)
    frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
    pause_rate = np.sum(np.mean(librosa.feature.rms(y=frames), axis=1) < 0.01) / frames.shape[1]

    # 11. Spectral Features (5)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # Create a dictionary to hold the features
    features = {
        'Energy': energy,
        'Zero Crossing Rate': zcr,
        'F0': f0,
        'F2': f2,
        'Jitter': jitter,
        'Shimmer': shimmer,
        'Band Energy Ratio': band_energy,
        'Pause Rate': pause_rate,
        'Spectral Centroid': centroid,
        'Spectral Bandwidth': bandwidth,
        'Spectral Rolloff': rolloff,
        'Spectral Flux': flux,
        'Spectral Flatness': flatness
    }

    # Convert lists to separate feature entries
    mfccs_mean = mfccs.mean(axis=1)
    chroma_mean = chroma.mean(axis=1)
    spectral_contrast_mean = spectral_contrast.mean(axis=1)

    # Add MFCC, Chroma, and Spectral Contrast features
    for i, mfcc in enumerate(mfccs_mean, start=1):
        features[f'MFCC_{i}'] = mfcc
    for i, chroma_val in enumerate(chroma_mean, start=1):
        features[f'Chroma_{i}'] = chroma_val
    for i, contrast in enumerate(spectral_contrast_mean, start=1):
        features[f'Spectral_Contrast_{i}'] = contrast

    return pd.DataFrame([features])

def predict_emotion(file_path, model):
    """Predict the emotion of a new audio file."""
    features_df = extract_features_for_prediction(file_path)
    prediction = model.predict(features_df)
    return prediction[0]
