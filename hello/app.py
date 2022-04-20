
from flask import Flask, render_template, request
import numpy as np
import pickle
#import xgboost as xgb
import subprocess

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
#model = xgb.Booster()
#model.load_model("model.pkl")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():


    gl = None
    mois = None
    day = None
    dayofyear = None
    dayofweek = None
    is_month_start = None
    is_month_end = None
    is_wknd = None
    daily_avg = None
    montly_avg = None

    gl = request.form.get('gl')
    mois = request.form.get('mois')
    day = request.form.get('day')
    dayofyear = request.form.get('dayofyear')
    dayofweek = request.form.get('dayofweek')
    is_month_start = request.form.get('is_month_start')
    is_month_end = request.form.get('is_month_end')
    is_wknd = request.form.get('is_wknd')
    daily_avg = request.form.get('daily_avg')
    montly_avg = request.form.get('montly_avg')

    arr = np.array([])
    arr = np.array([gl, mois, day, dayofyear, dayofweek, is_month_start, is_month_end, is_wknd, daily_avg, montly_avg])

    arr = arr.astype(np.float64)
    pred = model.predict([arr])
    data=int(pred)

    return str(data)


if __name__ == '__main__':
    app.run(debug=True)