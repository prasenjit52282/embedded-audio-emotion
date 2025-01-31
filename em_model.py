import emlearn
import sys
import joblib

#python em_model.py ./models/RandomForestClassifier
#will serialize the model to a embedded compatible header file (tested for randomforest model)

if __name__=='__main__':
    model_path = sys.argv[1]
    model = joblib.load(model_path)

    cmodel = emlearn.convert(model, method='inline')
    cmodel.save(file='classifier.h', name='rf')