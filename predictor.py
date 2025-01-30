import extractor
import sys
import joblib

def predict(file_path,model_path):
    try:
        # Predict the emotion using the saved file
        model = joblib.load(model_path)
        emotion = extractor.predict_emotion(file_path,model)

        print(emotion)
    except:
        print('An exception occured')

if __name__=='__main__':
    audio_file_path = sys.argv[1]
    model_path = sys.argv[2]
    predict(audio_file_path,model_path)