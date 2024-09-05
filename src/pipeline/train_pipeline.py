import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils import save_object, load_object

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def initiate_data_ingestion(self):
        try:
            return self.data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            return self.data_transformation.initiate_data_transformation(train_path, test_path)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            return self.model_trainer.initiate_model_trainer(train_arr, test_arr)
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Starting the training pipeline")

            train_data_path, test_data_path = self.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Train path: {train_data_path}, Test path: {test_data_path}")

            train_arr, test_arr, preprocessor_path = self.initiate_data_transformation(train_data_path, test_data_path)
            logging.info("Data transformation completed")

            r2_score = self.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model training completed. R2 score: {r2_score}")

            logging.info("Training pipeline completed successfully")
            return r2_score

        except Exception as e:
            logging.error("An error occurred in the training pipeline")
            raise CustomException(e, sys)

class DataInfo:
    def __init__(self, 
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()