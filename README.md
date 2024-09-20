# Customer Purchase Prediction Web Application

This project is a web application that predicts customer purchase amounts based on various features. It includes data ingestion, transformation, model training, and a Flask web interface for making predictions and retraining the model.

## Features

- Predict customer purchase amounts based on age, annual income, loyalty score, region, and purchase frequency
- Web interface for easy input and prediction
- Model retraining functionality
- Display of current model performance (R2 score)

## Project Structure

- `app.py`: Main Flask application
- `templates/home.html`: HTML template for the web interface
- `src/pipeline/`:
  - `predict_pipeline.py`: Handles prediction logic
  - `train_pipeline.py`: Manages the model training pipeline
- `src/components/`:
  - `data_ingestion.py`: Handles data loading and splitting
  - `data_transformation.py`: Preprocesses the data
  - `model_trainer.py`: Trains and evaluates multiple models

## Setup and Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure the dataset `Customer_Purchasing_Behaviors.csv` is in the `notebook/data/` directory

## Running the Application

1. Start the Flask server:
   ```
   python app.py
   ```
2. Open a web browser and navigate to `http://localhost:5000`

## Usage

1. Enter customer details in the form on the home page
2. Click "Predict Purchase Amount" to get a prediction
3. Use the "Retrain Model" button to retrain the model with the latest data
4. View the current model performance (R2 score) at the bottom of the page

## Model Training

The application uses various regression models including Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBoost, CatBoost, and AdaBoost. The best performing model is selected based on the R2 score.

## Data Preprocessing

The data transformation pipeline handles:
- Imputation of missing values
- Scaling of numerical features
- One-hot encoding of categorical features

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push to your fork and submit a pull request

## License

[Specify your license here]