from flask import Flask, render_template,request
import pickle
import numpy as np
import pandas as pd

model = 'mushroom_model.pkl'
logistic = pickle.load(open(model,'rb'))

model_encode = 'mushroom_classify_encoder.pkl'
onehot = pickle.load(open(model_encode, 'rb'))



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        CapShape = str(request.form['cap-shape'])
        CapSurface = str(request.form['cap-surface'])
        CapColor = str(request.form['cap-color'])
        Bruises = str(request.form['bruises'])
        Odor = str(request.form['odor'])
        GillAttachment = str(request.form['gill-attachment'])
        GillSpacing = str(request.form['gill-spacing'])
        GillSize = str(request.form['gill-size'])
        GillColor = str(request.form['gill-color'])
        StalkShape = str(request.form['stalk-shape'])
        StalkRoot = str(request.form['stalk-root'])
        Stalk_s_a_r = str(request.form['stalk-surface-above-ring'])
        Stalk_s_b_r = str(request.form['stalk-surface-below-ring'])
        Stalk_c_a_r = str(request.form['stalk-color-above-ring'])
        Stalk_c_b_r = str(request.form['stalk-color-below-ring'])
        Veil_type = str(request.form['veil-type'])
        Veil_color = str(request.form['veil-color'])
        Ring_num = str(request.form['ring-number'])
        Ring_type = str(request.form['ring-type'])
        Spore_P_C = str(request.form['spore-print-color'])
        Population = str(request.form['population'])
        Habit = str(request.form['habitat'])
        
        data =[[CapShape, CapSurface, CapColor, Bruises, Odor, GillAttachment,
                         GillSpacing, GillSize, GillColor,StalkShape, StalkRoot, Stalk_s_a_r, Stalk_s_b_r,
                         Stalk_c_a_r, Stalk_c_b_r, Veil_type, Veil_color, Ring_num, Ring_type,
                         Spore_P_C, Population, Habit]]
        
        onehotencode=onehot.transform(data)
        
        my_prediction = logistic.predict(onehotencode)
        
        return render_template('result.html',prediction=my_prediction)
    
if __name__=='__main__':
    app.run(debug=True,use_reloader=False)
        
