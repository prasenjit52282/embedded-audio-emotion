import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings
import sys
import os

MODELS_DIRECTORY = "models"
# Suppress warnings
warnings.filterwarnings("ignore")

class ModelEvaluator:
    def __init__(self, data_path, target_column, best_models=None, test_size=0.3, random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.df = pd.read_csv(data_path).dropna()
        self.X = self.df.drop(target_column, axis=1)
        self.y = self.df[target_column]
        
        self.best_models = best_models or [
            "ExtraTreesClassifier", "RandomForestClassifier", "BaggingClassifier",
            "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis", 
            "RidgeClassifier", "DecisionTreeClassifier"
        ]
        self.results = []
        self.all_models = []
        
        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
    def train_models(self):
        all_classifiers = all_estimators(type_filter="classifier")
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        # print(all_classifiers)
        os.makedirs(MODELS_DIRECTORY, mode=0o777, exist_ok=False)
        for name, Classifier in all_classifiers:
            if name not in self.best_models:
                continue
            try:
                print(name)
                model = Classifier()
                self._cross_validate_and_evaluate(model, name, kf)
                
            except Exception as e:
                print(f"Skipping {name} due to error: {e}")
    
    def _cross_validate_and_evaluate(self, model, model_name, kf):
        # Cross-validation on training set
        accuracy_scores = cross_val_score(model, self.X_train, self.y_train, cv=kf, scoring='accuracy')
        f1_scores = cross_val_score(model, self.X_train, self.y_train, cv=kf, scoring='f1_weighted')
        precision_scores = cross_val_score(model, self.X_train, self.y_train, cv=kf, scoring='precision_weighted')
        recall_scores = cross_val_score(model, self.X_train, self.y_train, cv=kf, scoring='recall_weighted')
        
        # Train the model and make predictions
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.all_models.append(model)
        self.save_model(model,f'{MODELS_DIRECTORY}/'+model_name)
        # Confusion matrix
        # self._plot_confusion_matrix(model, model_name, y_pred)
        
        # Test metrics
        test_accuracy_scores = cross_val_score(model, self.X_test, self.y_test, cv=kf, scoring='accuracy')
        test_accuracy_mean = test_accuracy_scores.mean()
        test_accuracy_std = test_accuracy_scores.std()
        test_f1 = f1_score(self.y_test, y_pred, average='weighted')
        test_precision = precision_score(self.y_test, y_pred, average='weighted')
        test_recall = recall_score(self.y_test, y_pred, average='weighted')
        
        # Store the results
        self.results.append({
            "Model": model_name,
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
        print(f'{model_name}: Train Acc {accuracy_scores.mean():.3f}, Test Acc {test_accuracy_mean:.3f}, Test Std {test_accuracy_std:.3f}')
    
    def _plot_confusion_matrix(self, model, model_name, y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def get_results(self):
        # Display the results as a DataFrame
        return pd.DataFrame(self.results)

    def save_model(self, model, filename):
        # Save the model to a file
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        # Load a model from a file
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model

if __name__=='__main__':
    data_path = sys.argv[1]
    me = ModelEvaluator(data_path,'Emotion')
    me.train_models()