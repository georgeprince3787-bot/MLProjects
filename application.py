from flask import Flask, request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score', 0)),
            writing_score=int(request.form.get('writing_score', 0))
        )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        # Print submitted inputs to terminal (so they appear in the server console)
        try:
            print("Submitted inputs:")
            print(pred_df.to_string(index=False))
            print(pred_df.shape)
        except Exception:
            # fallback: print raw form values
            print("Submitted inputs (fallback):", dict(request.form))

        # Render the same form page with the prediction and preserve submitted values
        return render_template(
            'home.html',
            prediction=results[0],
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score',''),
            writing_score=request.form.get('writing_score','')
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0")
