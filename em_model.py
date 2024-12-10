#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df_ = pd.read_csv("extracted_acoustic_features.csv").dropna()
df_.head()


# In[ ]:


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
rf_classifier = RandomForestClassifier(n_estimators=20,max_depth=30,random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[7]:


import emlearn
cmodel = emlearn.convert(rf_classifier, method='inline')
cmodel.save(file='classifier.h', name='rf')

