#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

from IPython.display import Audio

import zipfile
import os


extract_dirs = ['/kaggle/input/speech-emotion-recognition-en/Crema', '/kaggle/input/speech-emotion-recognition-en/Ravdess',
                '/kaggle/input/speech-emotion-recognition-en/Savee', '/kaggle/input/speech-emotion-recognition-en/Tess']



# In[ ]:


get_ipython().system('cp -r /kaggle/input/speech-emotion-recognition-en /kaggle/working/')


# In[ ]:


Root_dir = 'speech-emotion-recognition-en/'
Crema_path = Root_dir + "/Crema/"
Ravdess_path = Root_dir + "/Ravdess/audio_speech_actors_01-24/"
Savee_path = Root_dir + "/Savee/"
Tess_path = Root_dir + "/Tess/"
Crema_dir_list = os.listdir(Crema_path)
Ravdess_dir_list = os.listdir(Ravdess_path)
Savee_dir_list = os.listdir(Savee_path)
Tess_dir_list = os.listdir(Tess_path)


# # Crema dataset

# In[ ]:


emotions_crema = []
paths_crema = []

for it in Crema_dir_list:
    # storing file paths
    paths_crema.append(Crema_path + it)
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

emotions_crema_df = pd.DataFrame(emotions_crema, columns=['Emotions'])


path_crema_df = pd.DataFrame(paths_crema, columns=['Path'])
Crema_df = pd.concat([emotions_crema_df, path_crema_df], axis=1)
print(Crema_df.shape)
Crema_df.head()


# In[ ]:


emotion_counts = Crema_df['Emotions'].value_counts()
print(emotion_counts)


# # Ravdess dataset

# In[ ]:


emotions_ravdess = []
path_ravdess = []

for it in Ravdess_dir_list:
    actor = os.listdir(Ravdess_path + it)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        emotions_ravdess.append(int(part[2]))
        path_ravdess.append(Ravdess_path + it + '/' + file)


emotion_ravdess_df = pd.DataFrame(emotions_ravdess, columns=['Emotions'])

path_ravdess_df = pd.DataFrame(path_ravdess, columns=['Path'])
Ravdess_df = pd.concat([emotion_ravdess_df, path_ravdess_df], axis=1)

Ravdess_df.Emotions.replace({1:'neutral', 2:'calm',
                             3:'happy', 4:'sad', 5:'angry',
                             6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_df.head()


# In[ ]:


emotion_counts = Ravdess_df['Emotions'].value_counts()
print(emotion_counts)


# # Savee dataset
# 

# In[ ]:


emotions_savee = []
path_savee = []

for it in Savee_dir_list:
    path_savee.append(Savee_path + it)
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

emotion_savee_df = pd.DataFrame(emotions_savee, columns=['Emotions'])

path_savee_df = pd.DataFrame(path_savee, columns=['Path'])
Savee_df = pd.concat([emotion_savee_df, path_savee_df], axis=1)
Savee_df.head()


# In[ ]:


emotion_counts = Savee_df['Emotions'].value_counts()
print(emotion_counts)


# # Tess datset

# In[ ]:


emotions_tess = []
path_tess = []

for it in Tess_dir_list:
    directories = os.listdir(Tess_path + '/' + it)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[2]
        if part=='ps':
            emotions_tess.append('surprise')
        else:
            emotions_tess.append(part)
        path_tess.append(Tess_path + it + '/' + file)



emotion_tess_df = pd.DataFrame(emotions_tess, columns=['Emotions'])
path_tess_df = pd.DataFrame(path_tess, columns=['Path'])
Tess_df = pd.concat([emotion_tess_df, path_tess_df], axis=1)
Tess_df.head()


# In[ ]:


emotion_counts = Tess_df['Emotions'].value_counts()
print(emotion_counts)


# # Making the ultimate combined dataset

# In[ ]:


Sum_df = pd.concat([Crema_df, Ravdess_df, Savee_df, Tess_df], axis=0).reset_index(drop=True)
Sum_df.to_csv("Sum_df.csv",index=False)
Sum_df.head()


# In[ ]:


Sum_df["Emotions"].unique() # happy = happiness, sad = sadness, calm = neutral


# In[ ]:


Sum_df_rep = Sum_df.copy()
Sum_df_rep['Emotions'] = Sum_df_rep['Emotions'].replace({
    'happiness': 'happy',
    'sadness': 'sad',
    'calm': 'neutral'
})
print(Sum_df_rep.head())


# In[ ]:


emotion_counts = Sum_df_rep['Emotions'].value_counts()
print(emotion_counts)


# In[ ]:


Sum_df_rep.to_csv('audio_dataframe.csv', index=False)

print(Sum_df_rep["Path"][0])
print(Sum_df_rep["Emotions"][0])


# # Acustoic features

# In[ ]:


import pandas as pd
# num_files_per_emotion = 33
selected_files = []

# Loop through each unique emotion and select files
for emotion in Sum_df_rep['Emotions'].unique():
    # Filter DataFrame for the current emotion
    emotion_files = Sum_df_rep[Sum_df_rep['Emotions'] == emotion]

    # Shuffle the files
    emotion_files = emotion_files.sample(frac=1).reset_index(drop=True)  # Shuffle

    # Select the specified number of files
    selected_emotion_files = emotion_files.copy()#.head(num_files_per_emotion)

    # Add selected files to the list
    selected_files.append(selected_emotion_files)

# Concatenate selected files into a new DataFrame
# final_selection = pd.concat(selected_files, ignore_index=True)

# # Shuffle the final selection to mix emotions
# final_selection = final_selection.sample(frac=1).reset_index(drop=True)

final_selection=Sum_df_rep

# Display the final selected DataFrame
print(f"Total selected files: {len(final_selection)}")
print(final_selection)


# In[ ]:


import pandas as pd
import librosa
import numpy as np

df = final_selection.copy()

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

features_list = []

for index, row in df.iterrows():
    file_path = row['Path']
    emotion = row['Emotions']

    # Extract features
    features = extract_acoustic_features(file_path)
    features['Emotion'] = emotion  # Add the emotion to the features
    features_list.append(features)

features_df = pd.DataFrame(features_list)
print(features_df)


# In[ ]:


features_df.head()


# In[ ]:


mfcc_columns = pd.DataFrame(features_df['MFCC'].tolist(), columns=[f'MFCC_{i+1}' for i in range(13)])
chroma_columns = pd.DataFrame(features_df['Chroma'].tolist(), columns=[f'Chroma_{i+1}' for i in range(12)])
spectral_columns = pd.DataFrame(features_df['Spectral Contrast'].tolist(), columns=[f'Spectral_Contrast_{i+1}' for i in range(7)])

df_flattened = pd.concat([features_df.drop(['MFCC', 'Chroma', 'Spectral Contrast'], axis=1),
                           mfcc_columns,
                           chroma_columns,
                           spectral_columns], axis=1)


# In[ ]:


df_flattened.to_csv("extracted_acoustic_features.csv",index=False)


# # Training a model

# In[1]:


import pandas as pd
df_ = pd.read_csv("/kaggle/input/acoustic-features-dataset/extracted_acoustic_features.csv")
df_.head()


# In[2]:


df_ = df_.dropna()


# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # Import joblib for saving the model

# Define features and target
df = df_.copy()
X = df.drop('Emotion', axis=1)  # Features
y = df['Emotion']               # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model to a file
joblib.dump(rf_classifier, 'rf_classifier_model.joblib')
print("Model saved as 'rf_classifier_model.joblib'")


# In[13]:


from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
import warnings

# Define features and target
df = df_.copy()
X = df.drop('Emotion', axis=1)  # Features
y = df['Emotion']               # Target variable

# Ignore warnings for models that might throw them
warnings.filterwarnings("ignore")

# Get all classifier models from scikit-learn
all_classifiers = all_estimators(type_filter="classifier")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dictionary to store results
results = {}

# Loop through all classifier models
for name, Classifier in all_classifiers:
    try:
        # Initialize the model
        model = Classifier()
        
        # Train the model on the training data
        model.fit(X_train, y_train)
        
        # Make predictions on the test data
        y_pred = model.predict(X_test)
        
        # Calculate accuracy and store in results dictionary
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f'{name}: {accuracy:.3f}')
        
    except (ValueError, NotFittedError, TypeError) as e:
        # Skip models that aren't compatible or need specific parameters
        print(f'{name} could not be evaluated: {e}')

# Display overall results summary
print("\nModel Performance Summary:")
for model_name, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
    print(f"{model_name}: {score:.3f}")


# In[ ]:


import pandas as pd
import numpy as np
import librosa

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

# Usage:
emotion = predict_emotion('speech-emotion-recognition-en//Crema/1058_TIE_SAD_XX.wav', rf_classifier)
print("Predicted Emotion:", emotion)


# # Latest

# In[4]:


from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.datasets import make_classification
import pandas as pd
import warnings

df = df_.copy()
# Define features and target (assuming `df` is already defined as you mentioned)
X = df.drop('Emotion', axis=1)  # Features
y = df['Emotion']               # Target variable

# Ignore warnings for models that might throw them
warnings.filterwarnings("ignore")

# Get all classifier models from scikit-learn
all_classifiers = all_estimators(type_filter="classifier")

# Dictionary to store results
results = []

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Stratified K-Fold cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Loop through all classifier models
for name, Classifier in all_classifiers:
    try:
        # Initialize the model
        model = Classifier()
        
        # Cross-validation metrics
        accuracy_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        f1_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='f1_weighted')
        precision_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='precision_weighted')
        recall_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='recall_weighted')
        
        # Fit and evaluate on test set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')

        # Append results
        results.append({
            "Model": name,
            "Train Accuracy (Mean)": accuracy_scores.mean(),
            "Train Accuracy (Std)": accuracy_scores.std(),
            "Train F1-score": f1_scores.mean(),
            "Train Precision": precision_scores.mean(),
            "Train Recall": recall_scores.mean(),
            "Test Accuracy": test_accuracy,
            "Test F1-score": test_f1,
            "Test Precision": test_precision,
            "Test Recall": test_recall
        })

        print(f'{name}: Train Acc {accuracy_scores.mean():.3f}, Test Acc {test_accuracy:.3f}')

    except:
        # Skip models that aren't compatible or need specific parameters
        print('problem')
        continue

# Create a DataFrame for better visualization
results_df = pd.DataFrame(results)
print("\nSummary Table:")
print(results_df.sort_values(by="Test Accuracy", ascending=False).to_string(index=False))


# In[5]:


results_df.to_csv('result.csv', encoding='utf-8', index=False,mode='w',header= True)


# In[ ]:


from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.datasets import make_classification
import pandas as pd
import warnings
import numpy as np

df = df_.copy()
# Define features and target (assuming `df` is already defined)
X = df.drop('Emotion', axis=1)  # Features
y = df['Emotion']               # Target variable

# Ignore warnings for models that might throw them
warnings.filterwarnings("ignore")

# Get all classifier models from scikit-learn
all_classifiers = all_estimators(type_filter="classifier")

# Dictionary to store results
results = []

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Stratified K-Fold cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Loop through all classifier models
for name, Classifier in all_classifiers:
    try:
        # Initialize the model
        model = Classifier()
        
        # Cross-validation metrics on the training set
        accuracy_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        f1_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='f1_weighted')
        precision_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='precision_weighted')
        recall_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='recall_weighted')

        # Train the model and evaluate on the test set using cross_val_predict to get predictions
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics on the test set
        test_accuracy_scores = cross_val_score(model, X_test, y_test, cv=kf, scoring='accuracy')
        test_accuracy_mean = test_accuracy_scores.mean()
        test_accuracy_std = test_accuracy_scores.std()

        test_f1 = f1_score(y_test, y_pred, average='weighted')
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')

        # Append results
        results.append({
            "Model": name,
            "Train Accuracy (Mean)": accuracy_scores.mean(),
            "Train Accuracy (Std)": accuracy_scores.std(),
            "Train F1-score": f1_scores.mean(),
            "Train Precision": precision_scores.mean(),
            "Train Recall": recall_scores.mean(),
            "Test Accuracy (Mean)": test_accuracy_mean,
            "Test Accuracy (Std)": test_accuracy_std,
            "Test F1-score": test_f1,
            "Test Precision": test_precision,
            "Test Recall": test_recall
        })

        print(f'{name}: Train Acc {accuracy_scores.mean():.3f}, Test Acc {test_accuracy_mean:.3f}, Test Std {test_accuracy_std:.3f}')

    except:
        # Skip models that aren't compatible or need specific parameters
        continue

# Create a DataFrame for better visualization
results_df_util = pd.DataFrame(results)
print("\nSummary Table:")
print(results_df_util.sort_values(by="Test Accuracy (Mean)", ascending=False).to_string(index=False))


# In[ ]:


results_df_util.to_csv('result_mod.csv', encoding='utf-8', index=False,mode='w',header= True)


# In[6]:


print(type(all_classifiers))


# In[5]:


all_classifiers


# In[10]:





# In[8]:


from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.datasets import make_classification
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = df_.copy()
# Define features and target (assuming `df` is already defined)
X = df.drop('Emotion', axis=1)  # Features
y = df['Emotion']               # Target variable

# Ignore warnings for models that might throw them
warnings.filterwarnings("ignore")
all_classifiers = all_estimators(type_filter="classifier")
best_models = ["ExtraTreesClassifier","RandomForestClassifier","BaggingClassifier","LinearDiscriminantAnalysis","QuadraticDiscriminantAnalysis","RidgeClassifier","DecisionTreeClassifier"]

# Dictionary to store results
results_util = []
all_models = []
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Stratified K-Fold cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Loop through all classifier models
for name, Classifier in all_classifiers:
    try:
        # Initialize the model
        if name not in best_models:
            continue
        model = Classifier()
        
        # Cross-validation metrics on the training set
        accuracy_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        f1_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='f1_weighted')
        precision_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='precision_weighted')
        recall_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='recall_weighted')

        # Train the model and evaluate on the test set using cross_val_predict to get predictions
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_models.append(model)
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        # Calculate metrics on the test set
        test_accuracy_scores = cross_val_score(model, X_test, y_test, cv=kf, scoring='accuracy')
        test_accuracy_mean = test_accuracy_scores.mean()
        test_accuracy_std = test_accuracy_scores.std()

        test_f1 = f1_score(y_test, y_pred, average='weighted')
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')
        
        # Append results
        results_util.append({
            "Model": name,
            "Train Accuracy (Mean)": accuracy_scores.mean(),
            "Train Accuracy (Std)": accuracy_scores.std(),
            "Train F1-score": f1_scores.mean(),
            "Train Precision": precision_scores.mean(),
            "Train Recall": recall_scores.mean(),
            "Test Accuracy (Mean)": test_accuracy_mean,
            "Test Accuracy (Std)": test_accuracy_std,
            "Test F1-score": test_f1,
            "Test Precision": test_precision,
            "Test Recall": test_recall
        })

        print(f'{name}: Train Acc {accuracy_scores.mean():.3f}, Test Acc {test_accuracy_mean:.3f}, Test Std {test_accuracy_std:.3f}')
        
    except:
        # Skip models that aren't compatible or need specific parameters
        continue

# Create a DataFrame for better visualization
results_df_util = pd.DataFrame(results_util)
print("\nSummary Table:")
print(results_df_util)


# In[ ]:




