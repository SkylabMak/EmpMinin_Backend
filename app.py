from flask import Flask, render_template, request
from flask_restful import reqparse, Api
from werkzeug.exceptions import BadRequest
from flask import jsonify
import flask

import numpy as np
import pandas as pd
import ast

import os
import json

from model import predictEmp

curr_path = os.path.dirname(os.path.realpath(__file__))

feature_cols = ['department', 'education', 'recruitment_channel', 
                'no_of_trainings', 'previous_year_rating', 
                'length_of_service', 'awards_won', 
                'avg_training_score']

context_dict = {
    'feats': feature_cols,
    'zip': zip,
    'range': range,
    'len': len,
    'list': list,
}

app = Flask(__name__)
api = Api(app)

# # FOR FORM PARSING
# parser = reqparse.RequestParser()
# parser.add_argument('list', type=list)

# Define the parser and add the 'myList' argument
parser = reqparse.RequestParser()
parser.add_argument('list', type=list, location='json', required=True, help="List cannot be blank!")

@app.errorhandler(BadRequest)
def handle_bad_request(error):
    response = jsonify({'message': error.data['message']})
    response.status_code = 400
    return response

@app.route('/api/predict/json', methods=['POST'])
def api_predict_json():
    try:
        # Parse input JSON
        # data = flask.request.json
        # Parse input JSON
        data = request.json
        input_list = data['list']
        
        # Convert input list to DataFrame without considering the index
        input_df = pd.DataFrame([input_list], columns=feature_cols)
        
        print("data : ", input_df)
        
        # Predict
        y_pred = predictEmp(input_df)
        
        # Convert prediction to int and respond
        return jsonify({'message': "success", "pred":y_pred.item()})
    except Exception as e:
        print("error in API ->", str(e))
        return jsonify({'message': "error", "pred": "None", "e": str(e)}), 400

@app.route('/api/predict/jsonList', methods=['POST'])
def api_predict_json_list():
    try:
        # Extract the list of inputs from the request JSON
        input_list = request.json['list']
               
        # Initialize an empty list to store predictions
        predictions = []

        # Loop through each input in the input list
        for input_row in input_list:
            # Convert the current input row to a pandas DataFrame
            input_df = pd.DataFrame([input_row], columns=feature_cols)
            
            # Call predictEmp with the current input DataFrame
            prediction = predictEmp(input_df)
            
            # Append the prediction to the predictions list
            predictions.append(prediction.item())
        
        # Return the list of predictions as JSON
        return jsonify({'message': "success", "pred": predictions})
    except Exception as e:
        print("error in api_predict_json_list ->", str(e))
        return jsonify({'message': "error", 'error': str(e)}), 400
    
@app.route('/')
def index():
    return {'message':"use /api to use"}



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
