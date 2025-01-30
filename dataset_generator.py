import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

ROOT_DIR = 'speech-emotion-recognition-en/'
CREMA_PATH = ROOT_DIR + "/Crema/"
RAVDESS_PATH = ROOT_DIR + "/Ravdess/audio_speech_actors_01-24/"
SAVEE_PATH = ROOT_DIR + "/Savee/"
TESS_PATH = ROOT_DIR + "/Tess/"
CREMA_DIR_LIST = os.listdir(CREMA_PATH)
RAVDESS_DIR_LIST = os.listdir(RAVDESS_PATH)
SAVEE_DIR_LIST = os.listdir(SAVEE_PATH)
TESS_DIR_LIST = os.listdir(TESS_PATH)
GENERATED_DATASET_DIR = "generated_dataset/"
AUDIO_DATAFRAME_PATH = GENERATED_DATASET_DIR+"audio_dataframe.csv"
EXTRACTED_ACOUSTIC_FEATURES_PATH = GENERATED_DATASET_DIR+"extracted_acoustic_features.csv"

def crema_analysis():
    emotions_crema = []
    paths_crema = []

    for it in CREMA_DIR_LIST:
        # storing file paths
        paths_crema.append(CREMA_PATH + it)
        # storing file emotions
        part = it.split('_')
        if part[2] == 'SAD':
            emotions_crema.append('sad')
        elif part[2] == 'ANG':
            emotions_crema.append('angry')
        elif part[2] == 'DIS':
            emotions_crema.append('disgust')
        elif part[2] == 'FEA':
            emotions_crema.append('fear')
        elif part[2] == 'HAP':
            emotions_crema.append('happy')
        elif part[2] == 'NEU':
            emotions_crema.append('neutral')
        else:
            emotions_crema.append('Unknown')

    # dataframe for emotion of files
    emotions_crema_df = pd.DataFrame(emotions_crema, columns=['Emotions'])


    # dataframe for path of files.
    path_crema_df = pd.DataFrame(paths_crema, columns=['Path'])
    Crema_df = pd.concat([emotions_crema_df, path_crema_df], axis=1)

    return Crema_df

def ravedees_analysis():
    emotions_ravdess = []
    path_ravdess = []

    for it in RAVDESS_DIR_LIST:
        # There are 20 actors
        actor = os.listdir(RAVDESS_PATH + it)
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            # third part in each file represents the emotion associated to that file.
            emotions_ravdess.append(int(part[2]))
            path_ravdess.append(RAVDESS_PATH + it + '/' + file)


    emotion_ravdess_df = pd.DataFrame(emotions_ravdess, columns=['Emotions'])

    path_ravdess_df = pd.DataFrame(path_ravdess, columns=['Path'])
    Ravdess_df = pd.concat([emotion_ravdess_df, path_ravdess_df], axis=1)

    # changing integers to actual emotions.
    Ravdess_df['Emotions'] = Ravdess_df['Emotions'].replace({
    1: 'neutral', 
    2: 'calm',
    3: 'happy', 
    4: 'sad', 
    5: 'angry',
    6: 'fear', 
    7: 'disgust', 
    8: 'surprise'
    })


    return Ravdess_df

def savee_analysis():
    emotions_savee = []
    path_savee = []

    for it in SAVEE_DIR_LIST:
        path_savee.append(SAVEE_PATH + it)
        part = it.split('_')[1]
        part = part[:-6]
        if part == 'a':
            emotions_savee.append('angry')
        elif part == 'd':
            emotions_savee.append('disgust')
        elif part == 'f':
            emotions_savee.append('fear')
        elif part == 'h':
            emotions_savee.append('happiness')
        elif part == 'n':
            emotions_savee.append('neutral')
        elif part == 'sa':
            emotions_savee.append('sadness')
        elif part == 'su':
            emotions_savee.append('surprise')

        else:
            emotions_savee.append('Unknown')

    # dataframe for emotion of files
    emotion_savee_df = pd.DataFrame(emotions_savee, columns=['Emotions'])

    # dataframe for path of files.
    path_savee_df = pd.DataFrame(path_savee, columns=['Path'])
    Savee_df = pd.concat([emotion_savee_df, path_savee_df], axis=1)

    return Savee_df

def tess_analysis():
    emotions_tess = []
    path_tess = []

    for it in TESS_DIR_LIST:
        directories = os.listdir(TESS_PATH + '/' + it)
        for file in directories:
            part = file.split('.')[0]
            part = part.split('_')[2]
            if part=='ps':
                emotions_tess.append('surprise')
            else:
                emotions_tess.append(part)
            path_tess.append(TESS_PATH + it + '/' + file)



    # dataframe for emotion of files
    emotion_tess_df = pd.DataFrame(emotions_tess, columns=['Emotions'])

    # dataframe for path of files.
    path_tess_df = pd.DataFrame(path_tess, columns=['Path'])
    Tess_df = pd.concat([emotion_tess_df, path_tess_df], axis=1)
    
    return Tess_df
   

def create_audio_dataframe():
    Crema_df = crema_analysis()
    Ravdess_df = ravedees_analysis()
    Savee_df =  savee_analysis()
    Tess_df = tess_analysis()
    Sum_df = pd.concat([Crema_df, Ravdess_df, Savee_df, Tess_df], axis=0).reset_index(drop=True)

    # Assign the replaced values back to the 'Emotions' column
    Sum_df['Emotions'] = Sum_df['Emotions'].replace({
        'happiness': 'happy',
        'sadness': 'sad',
        'calm': 'neutral'
    })
    
    os.makedirs(GENERATED_DATASET_DIR, mode=0o777, exist_ok=False)

    # Save DataFrame to CSV
    Sum_df.to_csv(AUDIO_DATAFRAME_PATH, index=False)

def extract_acoustic_features(file_path):
    """Extract acoustic features from an audio file."""
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

    # 8. Jitter (1)
    jitter = np.mean(np.abs(np.diff(librosa.util.normalize(mfccs[0]))))

    # 9. Shimmer (1)
    shimmer = np.mean(np.abs(np.diff(y)))

    # 10. Band Energy Ratio (1)
    band_energy = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # 11. Pause Rate (1)
    frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
    pause_rate = np.sum(np.mean(librosa.feature.rms(y=frames), axis=1) < 0.01) / frames.shape[1]

    # 12. Spectral Features (5)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
    flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # Create a dictionary to hold the features
    features = {
        'MFCC': mfccs.mean(axis=1),  # Mean of each MFCC coefficient
        'Chroma': chroma.mean(axis=1),  # Mean of each chroma feature
        'Energy': energy,
        'Zero Crossing Rate': zcr,
        'Spectral Contrast': spectral_contrast.mean(axis=1),  # Mean of spectral contrast features
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

    return features

def acoustic_features_analysis_util():
    # Define the number of files to select per emotion
    num_files_per_emotion = 33  # Adjust as needed
    selected_files = []
    df = pd.read_csv(AUDIO_DATAFRAME_PATH)
    # Loop through each unique emotion and select files
    for emotion in df['Emotions'].unique():
        # Filter DataFrame for the current emotion
        emotion_files = df[df['Emotions'] == emotion]

        # Shuffle the files
        emotion_files = emotion_files.sample(frac=1).reset_index(drop=True)  # Shuffle

        # Select the specified number of files
        selected_emotion_files = emotion_files.head(num_files_per_emotion)

        # Add selected files to the list
        selected_files.append(selected_emotion_files)

    # Initialize a list to hold the features
    features_list = []

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        file_path = row['Path']
        emotion = row['Emotions']

        # Extract features
        features = extract_acoustic_features(file_path)
        features['Emotion'] = emotion  # Add the emotion to the features
        features_list.append(features)

    # Create a new DataFrame with the extracted features
    features_df = pd.DataFrame(features_list)

    return features_df

def acoustic_features_analysis():
    features_df = acoustic_features_analysis_util()
    mfcc_columns = pd.DataFrame(features_df['MFCC'].tolist(), columns=[f'MFCC_{i+1}' for i in range(13)])
    chroma_columns = pd.DataFrame(features_df['Chroma'].tolist(), columns=[f'Chroma_{i+1}' for i in range(12)])
    spectral_columns = pd.DataFrame(features_df['Spectral Contrast'].tolist(), columns=[f'Spectral_Contrast_{i+1}' for i in range(7)])

    # Concatenate these new DataFrame columns with the original DataFrame
    df_flattened = pd.concat([features_df.drop(['MFCC', 'Chroma', 'Spectral Contrast'], axis=1),
                            mfcc_columns,
                            chroma_columns,
                            spectral_columns], axis=1)
    df_flattened.to_csv(EXTRACTED_ACOUSTIC_FEATURES_PATH,index=False)

if __name__=='__main__':
    create_audio_dataframe()
    acoustic_features_analysis()