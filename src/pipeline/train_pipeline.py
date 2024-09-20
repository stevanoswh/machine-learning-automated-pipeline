import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting the training pipeline")

            # Step 1: Data Ingestion
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Train path: {train_data_path}, Test path: {test_data_path}")

            # Step 2: Data Transformation
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")

            # Step 3: Model Training
            r2_score = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model training completed. R2 score: {r2_score}")

            logging.info("Training pipeline completed successfully")
            return r2_score

        except Exception as e:
            logging.error("An error occurred during the training pipeline")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj=TrainPipeline()
    obj.run_pipeline()
    
