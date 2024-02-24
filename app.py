from flask import Flask, render_template, request
from flask_restful import reqparse, Api
from werkzeug.exceptions import BadRequest
from flask import jsonify
from flask_cors import CORS  # Import CORS
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
CORS(app)  # Enable CORS for all routes and origins
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
        # Extract the list of inputs from the request JSON, skipping the header
        full_input_list = request.json['list']
        header = full_input_list[0]  # This is your header row
        data_rows = full_input_list[1:]  # Skip the header row for processing
        # print("all row",data_rows)
        
        # Initialize an empty list to store predictions
        predictions = []
        header.append("is_promote")
        predictions.append(header)
        # Map header to indices for quick lookup
        header_to_index = {col_name: index for index, col_name in enumerate(header)}

        # Loop through each data row in the input list
        for input_row in data_rows:
            # print("current row raw -> ",input_row)
            # Create a list to hold the data for the required features
            selected_data = [input_row[header_to_index[col]] for col in feature_cols]
            
            # Convert the selected_data to a pandas DataFrame
            input_df = pd.DataFrame([selected_data], columns=feature_cols)
            # print("current row -> ",input_df)
            # Call predictEmp with the current input DataFrame
            prediction = predictEmp(input_df)
            # print("pred is ",prediction)
            # Append the prediction to the predictions list
            input_row.append(prediction.item())
            if prediction is not None:
                predictions.append(input_row)
            else:
                # Handle the error case, perhaps append a default value or log the error
                predictions.append(None)
        
        # Return the list of predictions as JSON
        return jsonify({'message': "success", "pred": predictions})
    except Exception as e:
        print("error in api_predict_json_list ->",str(e))
        return jsonify({'message': "error", 'error': str(e)}), 400


    
@app.route('/')
def index():
    return {'message':"use /api to use"}



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
