import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features
                train_array[:, -1],   # Target
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define the models to be trained
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Define hyperparameters for each model
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [10, 50, 100, 200],
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.01, 0.001],
                    'subsample': [0.8, 0.9, 1.0],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.01, 0.001],
                },
                "CatBoost Regressor": {
                    'iterations': [100, 200, 300],
                    'depth': [6, 8, 10],
                    'learning_rate': [0.1, 0.01, 0.001],
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.1, 0.01, 0.001],
                }
            }

            # Evaluate models and hyperparameters
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            logging.info(f"Model performance report: {model_report}")

            # Select the best model based on test R^2 score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.7:
                raise CustomException("No suitable model found with acceptable performance")

            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")

            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Best model saved at {self.model_trainer_config.trained_model_file_path}")

            # Test the best model on the test set
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
