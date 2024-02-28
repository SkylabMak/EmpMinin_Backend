import joblib 
import pandas as pd
import numpy as np
import os

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# load the model file
curr_path = os.path.dirname(os.path.realpath(__file__))
xgb_modelEmp = joblib.load(curr_path + "/model/xgb_model.joblib")
fitted_pipeline = joblib.load(curr_path + "/model/fitted_pipeline.joblib")
clustering  = joblib.load(curr_path + "/model/clustering_model.joblib")
scaler  = joblib.load(curr_path + "/model/scaler.joblib")


id_dep_columns = ["employee_id","is_promoted","gender","age","region"]
feature_cols = ['department', 'education', 'recruitment_channel', 
                'no_of_trainings', 'previous_year_rating', 
                'length_of_service', 'awards_won', 
                'avg_training_score','n_cluster']



def predictEmp(attributes: pd.DataFrame):
    try:
        print("att : ", attributes)
        
        # Ensure the 'attributes' is a DataFrame
        if not isinstance(attributes, pd.DataFrame):
            attributes = pd.DataFrame(attributes, columns=feature_cols) # Adjust 'feature_cols' as necessary
        
        # Select columns for clustering
        X_clus = attributes[["length_of_service", "avg_training_score", "awards_won"]]
        
        new_X_clus_scaled = scaler.transform(X_clus)
        
        # Apply clustering model to generate cluster labels
        n_cluster = clustering.predict(new_X_clus_scaled)
        
        # Add cluster labels to the attributes DataFrame
        attributes['n_cluster'] = n_cluster
        
        # Make sure to reorder / select columns as expected by the pipeline if necessary
        # Drop unnecessary columns if they are not expected by the fitted_pipeline
        attributes_for_pipeline = attributes.drop(columns=id_dep_columns, errors='ignore')
        
        # Apply the fitted pipeline transformations
        new_data_processed = fitted_pipeline.transform(attributes_for_pipeline)
        
        # Use the model to make predictions
        predictions = xgb_modelEmp.predict(new_data_processed)
        
        print("predictions are ", predictions)
        # Return the predictions
        return predictions
    except Exception as e:
        print("error in model ->", e)
        return None