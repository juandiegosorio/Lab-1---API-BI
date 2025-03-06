# Escenarios de Prueba para la API de Predicción de Precios de Viviendas

## 2.1. Escenarios de Prueba

### Escenario 1: Caso Exitoso - Predicción Coherente

**Datos de Entrada (JSON):**
```json
{
  "MedInc": 8.5,
  "HouseAge": 15.0,
  "AveRooms": 6.0,
  "AveBedrms": 2.0,
  "Population": 2000.0,
  "AveOccup": 3.0,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

**Resultado Esperado:**
Una predicción exitosa con un precio en el rango aproximado de $400,000 a $500,000.
![alt text](image-3.png)

**Análisis de Coherencia:**
Este escenario representa una vivienda en el área de la Bahía de San Francisco (según coordenadas) con valores razonables para todas las características. El ingreso medio es alto (8.5 = $85,000), la casa no es muy antigua (15 años), y la relación habitaciones/dormitorios es normal. La predicción resultante es coherente porque:

- Se encuentra dentro del rango de precios esperado para esa zona de California
- Los valores proporcionados son similares a los usados durante el entrenamiento
- La ubicación (Latitude/Longitude) corresponde a una zona de alto valor inmobiliario

### Escenario 2: Caso Exitoso - Valores Extremos Pero Válidos

**Datos de Entrada (JSON):**
```json
{
  "MedInc": 2.0,
  "HouseAge": 50.0,
  "AveRooms": 4.0,
  "AveBedrms": 1.5,
  "Population": 5000.0,
  "AveOccup": 4.5,
  "Latitude": 34.05,
  "Longitude": -118.24
}
```

**Resultado Esperado:**
Una predicción exitosa con un precio más bajo, probablemente en el rango de $150,000 a $250,000.
![alt text](image-2.png)

**Análisis de Coherencia:**
Este caso representa una vivienda más antigua en un área de menores ingresos cerca de Los Ángeles. La predicción debería ser coherente aunque menor que el Escenario 1 porque:

- El ingreso medio es bajo ($20,000)
- Las casas son significativamente más antiguas (50 años)
- El área está más densamente poblada
- La relación habitaciones/dormitorios sugiere espacios más pequeños

### Escenario 3: Caso de Error - Valores Fuera de Rango

**Datos de Entrada (JSON):**
```json
{
  "MedInc": 100.0,
  "HouseAge": 200.0,
  "AveRooms": 50.0,
  "AveBedrms": 25.0,
  "Population": 100000.0,
  "AveOccup": 30.0,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

**Resultado Esperado:**
La API devolverá una predicción, pero será incorrecta o extremadamente alta.
![alt text](image-1.png)

**Análisis de Incoherencia:**
Aunque el modelo probablemente no genere un error técnico y devolverá un valor, este será completamente incoherente porque:

- Los valores están muy fuera del rango de los datos de entrenamiento
- Es físicamente imposible tener casas con 200 años en California
- Un ingreso medio de $1,000,000 para un bloque es extremadamente atípico
- 50 habitaciones promedio por vivienda es irreal
- El modelo de regresión lineal puede extrapolar sin límites, generando predicciones absurdas

### Escenario 4: Caso de Error - Valores Faltantes

**Datos de Entrada (JSON):**
```json
{
  "MedInc": 8.5,
  "HouseAge": 15.0,
  "AveRooms": 6.0,
  "AveBedrms": 2.0,
  "Population": 2000.0,
  "AveOccup": 3.0,
  "Latitude": 37.88
}
```

**Resultado Esperado:**
Error de validación, ya que falta el campo "Longitude".
![alt text](image.png)

**Análisis del Error:**
Este escenario producirá un error de validación en FastAPI porque Pydantic detectará que falta un campo requerido. El modelo no llegará a ejecutarse porque la validación falla primero. Este es un comportamiento correcto que previene errores más graves durante la predicción.

## 2.2. Estrategia para Mitigar Incoherencias y Errores

Para mejorar la robustez de la API y reducir predicciones incoherentes, propongo implementar las siguientes estrategias:

1. **Validación avanzada de datos**: Añadir a la clase Pydantic validadores personalizados que verifiquen que los valores están dentro de rangos razonables (por ejemplo, ingresos entre 0-20, edades de casas entre 0-100, etc.), rechazando solicitudes con valores extremos antes de que lleguen al modelo.

2. **Detección de anomalías**: Implementar un componente que analice la distancia de Mahalanobis u otra métrica para detectar si un nuevo punto está demasiado alejado de los datos de entrenamiento, advirtiendo al usuario cuando las predicciones pueden ser poco fiables.

3. **Transformación y acotamiento de salidas**: Aplicar funciones de transformación que limiten las predicciones a rangos sensatos (por ejemplo, usando funciones sigmoid o tanh escaladas) para evitar valores absurdos incluso cuando los datos de entrada son extremos.

4. **Logging y monitoreo**: Registrar todas las predicciones junto con sus entradas para identificar patrones de uso problemáticos y mejorar continuamente el modelo y las validaciones.