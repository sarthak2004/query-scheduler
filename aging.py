import time
from flask import Flask
from flask_pymongo import PyMongo







app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"
db = PyMongo(app).db

while True:
    cursor = db.final_db.find()

    for document in cursor:
        new_priority = document['priority'] + 0.1
        customer_id = document['customer_id']
        product_id = document['product_id']
        db.final_db.update_one({'customer_id': customer_id, 'product_id': product_id}, {'$set': {'priority': new_priority}})
        
    time.sleep(300)

