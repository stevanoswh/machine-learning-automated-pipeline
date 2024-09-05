from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline

application = Flask(__name__)
app = application
app.debug = True

# Global variable to store the latest R2 score
latest_r2_score = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

@app.route('/retrain', methods=['POST'])
def retrain_model():
    global latest_r2_score
    train_pipeline = TrainPipeline()
    latest_r2_score = train_pipeline.run_pipeline()
    return jsonify({"message": "Model retrained successfully", "r2_score": latest_r2_score})

@app.route('/model_performance', methods=['GET'])
def get_model_performance():
    global latest_r2_score
    if latest_r2_score is None:
        return jsonify({"message": "Model has not been trained yet"})
    return jsonify({"r2_score": latest_r2_score})

if __name__ == "__main__":
    app.run(host="0.0.0.0")