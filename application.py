from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle as pk

# Load the trained model and preprocessor
model = pk.load(open(r"C:\Users\sajeel\Desktop\Data Science Complete\project1\artifacts\model.pkl", "rb"))
preprocess = pk.load(open(r"C:\Users\sajeel\Desktop\Data Science Complete\project1\artifacts\preprocess.pkl", "rb"))
df = pd.read_csv(r"C:\Users\sajeel\Desktop\Data Science Complete\project1\artifacts\clean_data.csv")

# Custom Functions 

screen_list = ["1920x1080","1366x768","1280x1024","1440x900","1600x900","1680x1050","1280x800","1024x768","2560x1440","3840x2160"]

def clean_ssd():
    ssd_list = []
    for i in df['ssd'].unique():
        ssd_list.append(round(i))
    new_list = sorted(ssd_list)
    return new_list

def clean_hdd():
    hdd_list = []
    for i in df['hdd'].unique():
        hdd_list.append(round(i))
    new_list2 = sorted(hdd_list)
    return new_list2

def screen_dpi(a, b):
    x = int(a.split("x")[0])
    y = int(a.split("x")[1])
    new_dpi = ((x)**2 + (y)**2)**0.5 / int(b)
    return new_dpi

# Flask App

application = Flask(__name__)
app = application

@app.route('/', methods=['GET', 'POST'])
def index():

    prediction_text = None
    input_data = None
    try:
        if request.method == 'POST':
            company = request.form['company']
            type_name = request.form['TypeName']
            screensize = request.form['screensize']
            cpu = request.form['Cpu']
            ram = request.form['Ram']
            Gpu = request.form['Gpu']
            os = request.form['OS']
            weight = request.form['weight']
            ips = request.form['IPS']
            Touchscreen = request.form['Touchscreen']
            Resolution = request.form['Resolution']
            SSD = request.form['SSD']
            HDD = request.form['HDD']

            dpi = screen_dpi(Resolution, screensize)
            input_data = [company, type_name, cpu, ram, Gpu, os, weight, ips, Touchscreen, dpi, SSD, HDD]
            
            # Convert input data to DataFrame with correct column names
            input_df = pd.DataFrame([input_data], columns=['Company', 'TypeName', 'Cpu', 'Ram', 'Gpu', 'OpSys', 'weight', 'ips', 'touch', 'screen_dpi', 'ssd', 'hdd'])
            
            # Transform the data
            transformed_data = preprocess.transform(input_df)
            
            # Make prediction
            prediction = model.predict(transformed_data)
            prediction_text = round(np.exp(prediction[0]))

    except Exception as e:
        prediction_text = "Error in prediction: " + str(e)

    return render_template('index.html',
                           prediction_text=prediction_text, 
                           option1=df['Company'].unique(), 
                           option2=df['TypeName'].unique(),
                           option3=df['Cpu'].unique(),
                           option4=sorted(df['Ram'].unique()),
                           option5=df['Gpu'].unique(),
                           option6=df['OpSys'].unique(),
                           option7=clean_ssd(),
                           option8=clean_hdd(),
                           option9=sorted(screen_list),
                           option10=input_data)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
