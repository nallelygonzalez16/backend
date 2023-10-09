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







# Cargar el modelo al inicio del servidor
modeloBodyFat = joblib.load('modeloBodyFat.pkl')

@app.route('/predictBodyFat', methods=['POST'])
def predictBodyFat():
    try:
        # Obtener datos JSON del cuerpo de la solicitud
        data = request.json
        present_Price  = data['Abdomen']
        Kms_Driven = data['Chest']

        # Convertir los datos a un array de NumPy
        input_data = np.array([[present_Price , Kms_Driven ]])

        # Realizar la predicción utilizando el modelo cargado
        prediction = modeloBodyFat.predict(input_data)

        # Devolver la predicción como respuesta JSON
        return jsonify({"Prediccion": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)})






# Cargar el modelo desde el archivo .pkl al inicio del servidor
modeloDelayVuelos = joblib.load('modeloDelayVuelos.pkl')

@app.route('/predictFlightDelay', methods=['POST'])
def predictFlightDelay():
    try:
        input_delay = ['DepDelay']  # Nombre de la variable que el modelo espera

        # Realizar la predicción utilizando el modelo cargado
        predicted_arrival_delay = modeloDelayVuelos.predict([[input_delay]])

        # Devolver la predicción como respuesta JSON
        return jsonify({"Prediction": predicted_arrival_delay[0]})
        

    except Exception as e:
        return jsonify({"error": str(e)})








if __name__ == '__main__':
    app.run(debug=True)