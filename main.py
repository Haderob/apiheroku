# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 02:47:08 2023

@author: Bese
"""
# fastaoi for creating api
from fastapi import FastAPI 
from fastapi.middleware.cors import CORMiddleware
from pydantic import BaseModel
# the format in which the input data is given to our mode
import pickle
# to change the jason data to dictionary
import json


app = FastAPI() # we will call this 'app' on our terminal

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
     
    Pregnancies :int
    Glucose :int
    BloodPressure : int
    SkinThickness: int
    Insulin :int
    BMI :float
    DiabetesPedigreeFunction:float
    Age : int
    
# loading the training model
diabetes_model=pickle.load(open('trained_model.sav','rb')) # if the darictory of the app.py and and the trained model are the same we don't need to specify the location here

@app.post('/diabetes_prediction')
def diabetes_predd(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    preg = input_dictionary['Pregnancies']
    glu = input_dictionary['Glucose']
    bp = input_dictionary['BloodPressure']
    skin = input_dictionary['SkinThickness']
    insulin = input_dictionary['Insulin']
    bmi = input_dictionary['BMI']
    dpf = input_dictionary['DiabetesPedigreeFunction']
    age = input_dictionary['Age']
    
    # creating a list of input
    input_list = [preg, glu, bp, skin, insulin, bmi, dpf, age]
    
    prediction = diabetes_model.predict([input_list]) # the list is inside [] if we dont do that we have to reshape the list to (1,-1)
    
    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'