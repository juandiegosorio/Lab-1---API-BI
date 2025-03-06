from typing import Optional, List
from fastapi import FastAPI, HTTPException
from joblib import load
import pandas as pd
from pydantic import BaseModel
import os

# Definir el modelo de datos Pydantic para las características de la vivienda
class DataModel(BaseModel):
    MedInc: float      # Ingreso medio del bloque (en decenas de miles)
    HouseAge: float    # Edad media de las casas del bloque
    AveRooms: float    # Número promedio de habitaciones por hogar
    AveBedrms: float   # Número promedio de dormitorios por hogar
    Population: float  # Población del bloque
    AveOccup: float    # Promedio de ocupantes por hogar
    Latitude: float    # Latitud del bloque
    Longitude: float   # Longitud del bloque
    
    # Esta función retorna los nombres de las columnas en el orden correcto
    def columns(self):
        return ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

# Modelo Pydantic para la respuesta
class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_dollars: float

# Inicializar la aplicación FastAPI
app = FastAPI(title="API de Predicción de Precios de Viviendas",
              description="API que predice el precio de viviendas basado en el dataset de California Housing",
              version="1.0.0")

# Ruta donde está guardado el modelo
MODEL_PATH = "assets/predictionPipeline.joblib"

# Variable global para almacenar el modelo cargado
modelo_info = None

@app.on_event("startup")
async def load_model():
    """Carga el modelo cuando la aplicación inicia"""
    global modelo_info
    try:
        modelo_info = load(MODEL_PATH)
        print(f"Modelo cargado correctamente desde {MODEL_PATH}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        # No lanzamos excepción aquí para permitir que la API inicie, 
        # pero los endpoints que usan el modelo verificarán si está cargado

@app.get("/")
def read_root():
    return {
        "mensaje": "API de Predicción de Precios de Viviendas",
        "descripción": "Utiliza el endpoint /predict para hacer predicciones",
        "ejemplo": {
            "MedInc": 8.5,
            "HouseAge": 15,
            "AveRooms": 6,
            "AveBedrms": 2,
            "Population": 2000,
            "AveOccup": 3,
            "Latitude": 37.88,
            "Longitude": -122.23
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def make_predictions(data: DataModel):
    """
    Realiza una predicción del precio de una vivienda
    
    - **MedInc**: Ingreso medio del bloque (en decenas de miles)
    - **HouseAge**: Edad media de las casas del bloque
    - **AveRooms**: Número promedio de habitaciones por hogar
    - **AveBedrms**: Número promedio de dormitorios por hogar
    - **Population**: Población del bloque
    - **AveOccup**: Promedio de ocupantes por hogar
    - **Latitude**: Latitud del bloque
    - **Longitude**: Longitud del bloque
    
    Devuelve el precio predicho en unidades de $100,000 y también en dólares
    """
    global modelo_info
    
    # Verificar si el modelo está cargado
    if modelo_info is None:
        try:
            modelo_info = load(MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"El modelo no está disponible: {str(e)}")
    
    try:
        # Extraer el pipeline del diccionario modelo_info
        pipeline = modelo_info['pipeline']
        
        # Convertir los datos a DataFrame y asegurar el orden correcto de las columnas
        df = pd.DataFrame(data.dict(), index=[0])
        df = df[data.columns()]
        
        # Realizar la predicción
        prediction = pipeline.predict(df)[0]
        
        # Formatear la respuesta (el precio está en unidades de $100,000)
        return {
            "predicted_price": float(prediction),
            "predicted_price_dollars": float(prediction * 100000)
        }
        
    except KeyError:
        # Si el modelo no tiene la estructura esperada (pipeline, feature_names, etc.)
        # Probablemente el modelo ha sido guardado de forma diferente
        try:
            # Intentar usar el modelo directamente si no está dentro de un diccionario
            prediction = modelo_info.predict(df)[0]
            return {
                "predicted_price": float(prediction),
                "predicted_price_dollars": float(prediction * 100000)
            }
        except Exception as e:
            raise HTTPException(status_code=500, 
                               detail=f"Error al procesar la predicción. El modelo podría tener un formato incorrecto: {str(e)}")
    
    except Exception as e:
        # Manejar cualquier otro error durante la predicción
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {str(e)}")