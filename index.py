from flask import Flask, request, jsonify
import joblib
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo previamente guardado
model = joblib.load('modelo_regresion_lineal.pkl')
@app.route('/home')
def home():
    return {"res":["API para modelo de regresión lineal"]}


model = joblib.load('modelo_regresion_lineal.pkl')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos JSON del cuerpo de la solicitud
        data = request.json
        tiempo_de_estudio = data['Tiempo_de_estudio']
        asistencia = data['Asistencia']
        # Convertir los datos a un array de NumPy
        input_data = np.array([[tiempo_de_estudio, asistencia]])
        # Realizar la predicción utilizando el modelo cargado
        prediction = model.predict(input_data)
        # Devolver la predicción como respuesta JSON
        return jsonify({"Prediccion": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})




#----------------------------------------------------MODELO 1-----------------------------------------------------
modeloBitcoin = joblib.load('modeloBitcoin.pkl')
@app.route('/predictBitcoin', methods=['POST'])
def predictBitcoin():
    try:
        # Obtener datos JSON del cuerpo de la solicitud
        data = request.json
        high_value = data['high_value']
        low_value = data['low_value']
        # Convertir los datos a un array de NumPy
        input_data = np.array([[high_value, low_value ]])
        # Realizar la predicción utilizando el modelo cargado
        prediction = modeloBitcoin.predict(input_data)
        # Devolver la predicción como respuesta JSON
        return jsonify({"Prediccion": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

#----------------------------------------------------MODELO 2-----------------------------------------------------
# Cargar el modelo al inicio del servidor
modeloCovid = joblib.load('modeloCovid.pkl')
@app.route('/predictCovid', methods=['POST'])
def predictCovid():
    try:
        # Obtener datos JSON del cuerpo de la solicitud
        data = request.json
        Confirmed  = data['Confirmed ']
        Deaths = data['Deaths']
        # Convertir los datos a un array de NumPy
        input_data = np.array([[Confirmed , Deaths ]])
        # Realizar la predicción utilizando el modelo cargado
        prediction = modeloCovid.predict(input_data)
        # Devolver la predicción como respuesta JSON
        return jsonify({"Prediccion": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

#----------------------------------------------------MODELO 3-----------------------------------------------------
# Cargar el modelo al inicio del servidor
modeloAutomovil = joblib.load('modeloAutomovil.pkl')
@app.route('/predictAutomovil', methods=['POST'])
def predictAutomovil():
    try:
        # Obtener datos JSON del cuerpo de la solicitud
        data = request.json
        present_Price  = data['present_Price ']
        Kms_Driven = data['Kms_Driven']
        # Convertir los datos a un array de NumPy
        input_data = np.array([[present_Price , Kms_Driven ]])
        # Realizar la predicción utilizando el modelo cargado
        prediction = modeloAutomovil.predict(input_data)
        # Devolver la predicción como respuesta JSON
        return jsonify({"Prediccion": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

#----------------------------------------------------MODELO 4-----------------------------------------------------
# Cargar el modelo al inicio del servidor
modeloBodyFat = joblib.load('modeloBodyFat.pkl')
@app.route('/predictBodyFat', methods=['POST'])
def predictBodyFat():
    try:
        # Obtener datos JSON del cuerpo de la solicitud
        data = request.json
        Abdomen  = data['Abdomen']
        Chest = data['Chest']
        # Convertir los datos a un array de NumPy
        input_data = np.array([[Abdomen , Chest ]])
        # Realizar la predicción utilizando el modelo cargado
        prediction = modeloBodyFat.predict(input_data)
        # Devolver la predicción como respuesta JSON
        return jsonify({"Prediccion": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

#----------------------------------------------------MODELO 5-----------------------------------------------------
# Cargar el modelo desde el archivo .pkl al inicio del servidor
modeloDelayVuelos = joblib.load('modeloDelayVuelos.pkl')
@app.route('/predictFlightDelay', methods=['POST'])
def predictFlightDelay():
    try:
        # Obtener datos JSON del cuerpo de la solicitud
        data = request.json
        LateAircraftDelay  = data['LateAircraftDelay']
        DepDelay = data['DepDelay']
        # Convertir los datos a un array de NumPy
        input_data = np.array([[LateAircraftDelay , DepDelay ]])
        # Realizar la predicción utilizando el modelo cargado
        prediction = modeloDelayVuelos.predict(input_data)
        # Devolver la predicción como respuesta JSON
        return jsonify({"Prediccion": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

#----------------------------------------------------MODELO 6-----------------------------------------------------
# Cargar el modelo desde el archivo .pkl al inicio del servidor
modeloAguacate = joblib.load('modeloAguacate.pkl')
@app.route('/predictAvocado', methods=['POST'])
def predictAvocado():
    try:
        # Obtener datos JSON del cuerpo de la solicitud
        data  = request.json
        year  = data['year']
        Total_Volume = data['Total_Volume']
        # Convertir los datos a un array de NumPy
        input_data = np.array([[year , Total_Volume ]])
        # Realizar la predicción utilizando el modelo cargado
        prediction = modeloAguacate.predict(input_data)
        # Devolver la predicción como respuesta JSON
        return jsonify({"Prediccion": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

#----------------------------------------------------MODELO 7-----------------------------------------------------


#----------------------------------------------------MODELO 8-----------------------------------------------------


#----------------------------------------------------MODELO 9-----------------------------------------------------


#----------------------------------------------------MODELO 10-----------------------------------------------------










if __name__ == '__main__':
    app.run(debug=True)