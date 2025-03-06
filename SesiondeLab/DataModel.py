from pydantic import BaseModel

class DataModel(BaseModel):
    # Estas variables permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    MedInc: float      # Ingreso medio del bloque (en decenas de miles)
    HouseAge: float    # Edad media de las casas del bloque
    AveRooms: float    # Número promedio de habitaciones por hogar
    AveBedrms: float   # Número promedio de dormitorios por hogar
    Population: float  # Población del bloque
    AveOccup: float    # Promedio de ocupantes por hogar
    Latitude: float    # Latitud del bloque
    Longitude: float   # Longitud del bloque
    
    # Esta función retorna los nombres de las columnas correspondientes con el modelo exportado en joblib.
    def columns(self):
        return ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]