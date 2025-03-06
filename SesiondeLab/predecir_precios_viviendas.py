import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import pickle
import os

def entrenar_y_guardar_pipeline(formato_guardado='joblib', ruta_guardado='modelo_precio_viviendas'):
    """
    Entrena un pipeline de regresión lineal para predecir precios de viviendas
    y lo guarda en formato .joblib o .pkl
    
    Parámetros:
    formato_guardado: str, 'joblib' o 'pkl' (por defecto 'joblib')
    ruta_guardado: str, ruta base del archivo sin extensión (por defecto 'modelo_precio_viviendas')
    
    Retorna:
    str: Ruta completa donde se guardó el modelo
    """
    # Cargar dataset de viviendas de California
    print("Cargando dataset de viviendas de California...")
    housing = fetch_california_housing()
    
    # Crear DataFrame con los datos
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    
    # Información sobre el dataset
    print(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"Características: {', '.join(housing.feature_names)}")
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Datos divididos: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} muestras de prueba")
    
    # Crear pipeline con escalado y regresión lineal
    print("Creando y entrenando pipeline...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Entrenar el modelo
    pipeline.fit(X_train, y_train)
    
    # Evaluar modelo
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Modelo entrenado. MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Guardar información de características para futuras predicciones
    modelo_info = {
        'pipeline': pipeline,
        'feature_names': housing.feature_names,
        'metrics': {'mse': mse, 'r2': r2},
        'coeficientes': pipeline.named_steps['regressor'].coef_
    }
    
    # Determinar la ruta de guardado completa
    if formato_guardado.lower() == 'joblib':
        ruta_completa = f"{ruta_guardado}.joblib"
        print(f"Guardando pipeline como archivo joblib: {ruta_completa}")
        joblib.dump(modelo_info, ruta_completa)
    elif formato_guardado.lower() == 'pkl':
        ruta_completa = f"{ruta_guardado}.pkl"
        print(f"Guardando pipeline como archivo pickle: {ruta_completa}")
        with open(ruta_completa, 'wb') as archivo:
            pickle.dump(modelo_info, archivo)
    else:
        raise ValueError("Formato de guardado no válido. Use 'joblib' o 'pkl'")
    
    print(f"Pipeline guardado exitosamente en: {os.path.abspath(ruta_completa)}")
    return os.path.abspath(ruta_completa)

def cargar_pipeline(ruta_archivo):
    """
    Carga un pipeline guardado anteriormente
    
    Parámetros:
    ruta_archivo: str, ruta completa al archivo .joblib o .pkl
    
    Retorna:
    dict: Diccionario con el pipeline y su información asociada
    """
    print(f"Cargando pipeline desde: {ruta_archivo}")
    
    if ruta_archivo.endswith('.joblib'):
        modelo_info = joblib.load(ruta_archivo)
    elif ruta_archivo.endswith('.pkl'):
        with open(ruta_archivo, 'rb') as archivo:
            modelo_info = pickle.load(archivo)
    else:
        raise ValueError("Formato de archivo no soportado. Debe ser .joblib o .pkl")
    
    print("Pipeline cargado exitosamente")
    print(f"Características del modelo: {', '.join(modelo_info['feature_names'])}")
    print(f"Métricas del modelo - MSE: {modelo_info['metrics']['mse']:.4f}, R²: {modelo_info['metrics']['r2']:.4f}")
    
    return modelo_info

def predecir_con_pipeline_guardado(ruta_archivo, datos_vivienda):
    """
    Realiza predicciones usando un pipeline guardado
    
    Parámetros:
    ruta_archivo: str, ruta al archivo .joblib o .pkl
    datos_vivienda: dict, diccionario con características de la vivienda
    
    Retorna:
    float: Precio predicho de la vivienda (en $100,000)
    """
    # Cargar el modelo
    modelo_info = cargar_pipeline(ruta_archivo)
    pipeline = modelo_info['pipeline']
    feature_names = modelo_info['feature_names']
    
    # Verificar que todas las características estén presentes
    for feature in feature_names:
        if feature not in datos_vivienda:
            raise ValueError(f"Falta la característica '{feature}' en los datos proporcionados")
    
    # Crear DataFrame con los datos en el orden correcto
    X_nuevo = pd.DataFrame([datos_vivienda], columns=feature_names)
    
    # Realizar predicción
    precio_predicho = pipeline.predict(X_nuevo)[0]
    
    # Mostrar resultado
    print(f"Precio predicho para la vivienda: ${precio_predicho*100000:.2f}")
    
    return precio_predicho

# Ejemplo de uso completo
if __name__ == "__main__":
    # 1. Entrenar y guardar el modelo
    formato = input("¿En qué formato desea guardar el modelo? (joblib/pkl): ").lower() or 'joblib'
    ruta_guardado = input("Ingrese el nombre del archivo (sin extensión): ") or 'modelo_precio_viviendas'
    
    ruta_completa = entrenar_y_guardar_pipeline(formato_guardado=formato, ruta_guardado=ruta_guardado)
    
    # 2. Ejemplo: Cargar el modelo y hacer una predicción
    print("\n--- Probando el modelo guardado ---")
    
    nueva_vivienda = {
        'MedInc': 8.5,           # Ingreso medio de 85,000
        'HouseAge': 15,          # Casas de 15 años en promedio
        'AveRooms': 6,           # Promedio de 6 habitaciones por hogar
        'AveBedrms': 2,          # Promedio de 2 dormitorios por hogar
        'Population': 2000,      # Población del bloque
        'AveOccup': 3,           # Promedio de 3 ocupantes por hogar
        'Latitude': 37.88,       # Latitud
        'Longitude': -122.23     # Longitud
    }
    
    # Realizar predicción con el modelo guardado
    precio = predecir_con_pipeline_guardado(ruta_completa, nueva_vivienda)