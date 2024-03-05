from flask import Flask, request, render_template
from flask_pymongo import PyMongo
import json
import time

import requests


app = Flask(__name__)
app.config["MONGO_URI"]="mongodb://localhost:27017/myDatabase"
db= PyMongo(app).db

def document_exists(customer_id, product_id):
    query = {"customer_id": customer_id, "product_id": product_id}
    result = db.inventory.find_one(query)
    return result is not None


@app.route("/",methods=['GET','POST'])

def index():
    if request.method=='POST':
        user_id= request.form['user']
        subject= request.form['subject']
        product_id= request.form['product']

        file_path = "assets/recording.wav"
        with open(file_path, "rb") as audio_file:
            audio_data = audio_file.read()


        document = {
            "customer_id": user_id,
            "subject": subject,
            "product_id": product_id,
            "timestamp": time.time(),
            "audio": audio_data,
            "flag": 1
        }

        existing_document = db.inventory.find_one({"customer_id": user_id, "product_id": product_id})
        if not existing_document:
            db.inventory.insert_one(document)
        return render_template('index.html')

      
    else:
        return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)