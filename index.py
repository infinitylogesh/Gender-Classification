from sklearn.externals import joblib
from flask import Flask,request,jsonify
import numpy as np
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24) #This is required
model = joblib.load("rf_gender_classification.pkl")

output = dict()

@app.route('/',methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        # return request.data
        body = request.data
        x_value = np.array([float(body.weight),float(body.height)])
        output["pred"] = model.predict(x_value)
        return jsonify(output)
