from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd
from logging import FileHandler,WARNING


app = Flask(__name__,template_folder='templates')
#app=application
file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)

scaler=pickle.load(open("templates/standardScaler.pkl", "rb"))
model = pickle.load(open("templates/modelForPrediction.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('/config/workspace/index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = result + 'Diabetic'
        else:
            result =result + 'Non-Diabetic'
            
        return render_template('/config/workspace/single_prediction.html',result=result)

    else:
        return render_template('/config/workspace/home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)