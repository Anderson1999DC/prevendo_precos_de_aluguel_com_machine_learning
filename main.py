from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Carregando modelo e colunas treinadas
modelo = joblib.load("modelo_aluguel_sp.pkl")
colunas = joblib.load("colunas_modelo.pkl")

app = FastAPI(
    title="Previsão de Aluguel de imóveis em SP",
    description="Modelo Random Forest treinado com dados de apartamentos de São Paulo (abril/2019)",
    version="1.0.0"
)

# Estrutura dos dados de entrada
class Imovel(BaseModel):
    Condo: float
    Size: float
    Rooms: int
    Toilets: int
    Suites: int
    Parking: int
    Elevator: int
    Furnished: int
    Swimming_Pool: int
    Latitude: float
    Longitude: float
    District: str

@app.get("/")
def root():
    return {"status": "online", "modelo": "Random Forest", "versao": "1.0.0"}

@app.post("/predict")
def predict(imovel: Imovel):
    # Montando dicionário com os dados recebidos
    dados = {
        "Condo": imovel.Condo,
        "Size": imovel.Size,
        "Rooms": imovel.Rooms,
        "Toilets": imovel.Toilets,
        "Suites": imovel.Suites,
        "Parking": imovel.Parking,
        "Elevator": imovel.Elevator,
        "Furnished": imovel.Furnished,
        "Swimming Pool": imovel.Swimming_Pool,
        "Latitude": imovel.Latitude,
        "Longitude": imovel.Longitude,
        "District": imovel.District
    }

    # Criando DataFrame e aplicando get_dummies igual ao treino
    df = pd.DataFrame([dados])
    df_encoded = pd.get_dummies(df, columns=["District"], dtype=int)

    # Alinhando colunas com o modelo treinado
    df_final = df_encoded.reindex(columns=colunas, fill_value=0)

    # Fazendo a previsão
    preco = modelo.predict(df_final)[0]

    return {
        "preco_previsto": round(float(preco), 2),
        "unidade": "BRL",
        "modelo": "RandomForestRegressor"
    }

from fastapi.responses import HTMLResponse

@app.get("/app", response_class=HTMLResponse)
def interface():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
