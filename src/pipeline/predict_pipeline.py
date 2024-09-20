import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Define paths to the preprocessor and model artifacts
        self.model_path = os.path.join("artifact", "model.pkl")
        self.preprocessor_path = os.path.join("artifact", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        try:
            # Load the saved model and preprocessor
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            # Apply preprocessing to the input features
            data_scaled = preprocessor.transform(features)

            # Make predictions
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, age: int, annual_income: float, loyalty_score: float, region: str, purchase_frequency: int):
        # Store user input data
        self.age = age
        self.annual_income = annual_income
        self.loyalty_score = loyalty_score
        self.region = region
        self.purchase_frequency = purchase_frequency

    def get_data_as_data_frame(self):
        try:
            # Convert user inputs to a pandas DataFrame
            custom_data_input_dict = {
                "age": [self.age],
                "annual_income": [self.annual_income],
                "loyalty_score": [self.loyalty_score],
                "region": [self.region],
                "purchase_frequency": [self.purchase_frequency],
            }

            # Return a DataFrame containing the input data
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
