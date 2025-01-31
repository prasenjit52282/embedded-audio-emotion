import warnings
import sys
import os
import joblib
import pandas as pd
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

plt.rcParams.update({
    'figure.figsize': (7,5),        # Set default figure size (width, height) in inches
    'axes.labelsize': 19,            # Font size for x and y labels
    'xtick.labelsize': 19,           # Font size for x-axis ticks
    'ytick.labelsize': 19,           # Font size for y-axis ticks
    'axes.labelweight': 'bold'
})

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted')
}

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
        self.df[target_column]=self.df[target_column].apply(str.capitalize)
        self.y = self.df[target_column]
        
        self.best_models = best_models or ["RandomForestClassifier", "ExtraTreesClassifier",
                                           "BaggingClassifier","QuadraticDiscriminantAnalysis",
                                           "LinearDiscriminantAnalysis","RidgeClassifier",
                                           "DecisionTreeClassifier"]
        all_classifiers = dict(all_estimators(type_filter="classifier"))
        self.selected_models={name:all_classifiers[name] for name in self.best_models}
        self.results = []
        self.all_models = []
        
        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
    def train_models(self):
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        # print(all_classifiers)
        os.makedirs(MODELS_DIRECTORY, mode=0o777, exist_ok=True)
        for name, Classifier in self.selected_models.items():
            np.random.seed(42)
            model = Classifier()

            # Cross-validation metrics on the training set
            x_val_scores=cross_validate(model, self.X, self.y, cv=kf,scoring=scoring,return_train_score=True)

            # Train the model and evaluate on the test set using cross_val_predict to get predictions
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_pred_train = model.predict(self.X_train)
            self.all_models.append(model)
            self.save_model(model,f'{MODELS_DIRECTORY}/'+name)

            cm = confusion_matrix(self.y_test, y_pred)

            # Plot the confusion matrix

            plt.figure()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_,
                        annot_kws={"size":19,})
            print(f'Confusion Matrix for {name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=30,horizontalalignment='right')
            plt.yticks(rotation=30,verticalalignment='top')

            plt.tight_layout()
            plt.savefig(f"./logs/matrix_{name}.pdf")

            # Calculate metrics on the test set
            #train
            train_f1 = f1_score(self.y_train, y_pred_train, average='weighted')
            train_precision = precision_score(self.y_train, y_pred_train, average='weighted')
            train_recall = recall_score(self.y_train, y_pred_train, average='weighted')

            #test
            test_f1 = f1_score(self.y_test, y_pred, average='weighted')
            test_precision = precision_score(self.y_test, y_pred, average='weighted')
            test_recall = recall_score(self.y_test, y_pred, average='weighted')

            # Append results
            self.results.append({
                "Model": name,
                "Train F1-score": round(train_f1,3),
                "Train Precision": round(train_precision,3),
                "Train Recall": round(train_recall,3),
                "Test F1-score": round(test_f1,3),
                "Test Precision": round(test_precision,3),
                "Test Recall": round(test_recall,3),
                "Train Accuracy (Mean)": round(x_val_scores['train_accuracy'].mean(),3),
                "Test Accuracy (Mean)": round(x_val_scores['test_accuracy'].mean(),3),
                "Train Accuracy (Std)": round(x_val_scores['train_accuracy'].std(),3),
                "Test Accuracy (Std)": round(x_val_scores['test_accuracy'].std(),3)
            })
    
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
    print("Saving performance results in logs...")
    me.get_results().to_csv("./logs/performance.csv")