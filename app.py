from flask import Flask, jsonify, request
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

import random
# load the trained model
model = pickle.load(open('model.pkl', 'rb'))
# load the doctor data

df = pd.read_csv('doctor.csv', encoding='ISO-8859-1')


# initialize the Flask app
app = Flask(__name__)

app.config['CORS_ENABLED'] = True

CORS(app)
@app.after_request
def add_cors_headers(response):
    # replace '*' with the URL of your frontend
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


# define the API endpoint
@app.route('/recommend', methods=['POST'])

def recommend():
    # parse the request data
    data = request.get_json()

    # extract the user inputs from the request data
    # location = data['location']
    # experience = data['experience']

    # convert location to numerical value
    # le = LabelEncoder()
    # location = le.fit_transform([location])[0]

    # get the recommended doctors
    # user = [0, experience, location]
    data = request.get_json(force=True)
    user = np.array(list(data.values()))

    distances, indices = model.kneighbors([user])
    recommended_doctors=[]
    experience = []
    contacts = []
    for i in indices[0]:
        recommended_doctors.append(df.iloc[i]['name'])
        experience.append(str(df.iloc[i]['experience']))
        phone = ''.join([str(random.randint(0, 9)) for i in range(10)])
        contacts.append(phone)
    # return the recommended doctors as a JSON response
    response = {"data":{'recommended_doctors': recommended_doctors,'experience': experience,"contacts":contacts}}
    return jsonify(response)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "doctor-recommendation-ml"})

# start the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
