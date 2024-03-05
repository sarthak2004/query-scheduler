from flask import Flask, request, render_template,redirect,url_for
from flask_pymongo import PyMongo
import pandas as pd
import numpy as np
import librosa
import whisper
import numpy as np
import pandas as pd
import nltk
import keras
import pickle
from keras.models import load_model
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import openai
import shutil
from keras.models import model_from_json
import requests
import random
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer




app = Flask(__name__)
app.config["MONGO_URI"]="mongodb://localhost:27017/myDatabase"
db= PyMongo(app).db








@app.route("/",methods=['GET','POST'])

def index():
    
    cursor = db.inventory.find()
    i = 0
    for document in cursor:
        audio_byte = document["audio"]
        with open("retrieved_audio.wav", "wb") as f:
            f.write(audio_byte)

        subject= [document["subject"]]
       

        # machine learning
            
        model_text = whisper.load_model("base")
        sia = SentimentIntensityAnalyzer()

        model = load_model("best_model1.keras")

        
        with open('scaler2.pickle', 'rb') as f:
            scaler2 = pickle.load(f)

        with open('encoder2.pickle', 'rb') as f:
            encoder2 = pickle.load(f)



        tfidf = joblib.load('tfidf.joblib')
        
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        class StemmedCountVectorizer(CountVectorizer):
            def build_analyzer(self):
                analyzer = super(StemmedCountVectorizer, self).build_analyzer()
                return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
        stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
        X_train_counts = stemmed_count_vect.fit_transform(subject)


        tfidf_transformer = tfidf
        X_train_tfidf= tfidf_transformer.transform(X_train_counts)

        model2= joblib.load("urgency_classifier.joblib")
        model2_train= model2.predict(X_train_tfidf)-1

        def zcr(data, frame_length, hop_length):
            zcr_values = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
            return np.squeeze(zcr_values)

        def rmse(data, frame_length=2048, hop_length=512):
            rmse_values = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
            return np.squeeze(rmse_values)

        def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
            mfcc_values = librosa.feature.mfcc(y=data, sr=sr, n_fft=frame_length, hop_length=hop_length)
            return np.squeeze(mfcc_values.T) if not flatten else np.ravel(mfcc_values.T)

        def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
            result = np.array([])

            result = np.hstack((result,
                                zcr(data, frame_length, hop_length),
                                rmse(data, frame_length, hop_length),
                                mfcc(data, sr, frame_length, hop_length)
                            ))
            return result

        def get_predict_feat(path):
            res=extract_features(path)
            result=np.array(res)
            desired_length = 2376
            if len(result) < desired_length:
                result = np.pad(result, (0, desired_length - len(result)), 'constant')
            result=np.reshape(result,newshape=(1,2376))
            i_result = scaler2.transform(result)
            final_result=np.expand_dims(result, axis=2)
            
            return final_result

        def prediction(path1):
            res=get_predict_feat(path1)
            predictions=model.predict(res)
            y_pred = encoder2.inverse_transform(predictions)
            return y_pred[0][0]

        def split_audio(audio_path, duration=2.5):
            y, sr = librosa.load(audio_path, sr=22050, offset = 0.6)
            samples_per_interval = int(duration * sr)
            audio_intervals = [y[i:i + samples_per_interval] for i in range(0, len(y), samples_per_interval)]
            return audio_intervals



        def get_scores(input_file):
            predictions_arr = []
            confidence_scores_arr = []
            audio_path = input_file
            intervals_array = split_audio(audio_path)
            for i in intervals_array:
                predicted_class = prediction(i)
                predictions_arr.append(predicted_class)

            for i in range(0, len(predictions_arr)):
                if predictions_arr[i] == 'sad':
                    predictions_arr[i] = 0.5
                if predictions_arr[i] == 'angry':
                    predictions_arr[i] = 1
                if predictions_arr[i] == 'happy':
                    predictions_arr[i] = -1
                if predictions_arr[i] == 'neutral':
                    predictions_arr[i] == -0.5

            for i in range(0, len(predictions_arr) // 2):
                predictions_arr[i] *= 0.8

            speech_score = sum(predictions_arr) / (((len(predictions_arr) // 2) * 0.8) + len(predictions_arr) - (len(predictions_arr) // 2))   
            
            result2 = model_text.transcribe(input_file , fp16=False)
            text_score = sia.polarity_scores(result2['text'])['compound']
            final_score = (-0.15 * text_score) + (0.35 * speech_score)+(0.5*model2_train)
            return final_score
        

        sound_file = "assets/recording.wav"
        final_score = get_scores(sound_file)








        # model end
        priority = final_score
        data = {
            "customer_id": document["customer_id"],
            "product_id": document["product_id"],
            "subject": document["subject"],
            "timestamp": document["timestamp"],
            "priority": priority
        }
        existing_document = db.final_db.find_one({"customer_id": document["customer_id"], "product_id": document["product_id"]})
        if not existing_document:
            db.final_db.insert_one(data)
        i += 1

    cursor2 = db.final_db.find()

    data_list = list(cursor2)
    sorted_data_list = sorted(data_list, key=lambda x: (-x["priority"], x["timestamp"]))

    return render_template('test.html', data_list=sorted_data_list)



@app.route('/remove_row', methods=['POST'])
def remove_row():
    if request.method == 'POST':
        customer_id = request.form['customer_id']
        product_id = request.form['product_id']

        db.inventory.delete_one({'customer_id': customer_id, 'product_id': product_id})
        db.final_db.delete_one({'customer_id': customer_id, 'product_id': product_id})


        return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True, port= 5001)