import json
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import pickle

#Comienzo de la api
#para levantar fast api: uvicorn main:app --reload
app = FastAPI()

#lectura del json y creación data frame
rows = []
with open('steam_games.json') as f: 
    rows.extend(ast.literal_eval(line) for line in f)
data_steam = pd.DataFrame(rows)


#Limpieza de data
data_steam['release_date'] = pd.to_datetime(data_steam['release_date'], errors='coerce')
specific_date = pd.to_datetime('1900-01-01')
data_steam['release_date'] = data_steam['release_date'].fillna(specific_date)
data_steam['metascore'] = pd.to_numeric(data_steam['metascore'], errors='coerce')
data_steam['price'] = pd.to_numeric(data_steam['price'], errors='coerce')

replacement_values = {'publisher': '', 'genres': '', 'tags': '', 'discount_price': 0, 'price': 0,
                      'specs': '', 'reviews_url': '', 'metascore': 0, 'app_name': '', 'title': '',
                       'id': '', 'sentiment': '', 'developer': ''}
data_steam.fillna(value=replacement_values, inplace=True)

# Retorna los 5 géneros más vendidos en el año indicado
@app.get('/genero/')
def genero(año: int):
    filtered_data_steam = data_steam[data_steam['release_date'].dt.year == año]
    # desanidar
    exploded_genres_data_steam = filtered_data_steam.explode('genres')
    top_genres = exploded_genres_data_steam['genres'].value_counts().nlargest(5).index.tolist()
    return top_genres

# Retorna juegos lanzados en el año indicado
@app.get('/juegos/')
def juegos(año: int):
    filtered_data_steam = data_steam[data_steam['release_date'].dt.year == año]
    released_games = filtered_data_steam['app_name'].tolist()
    return released_games

# Retorna 5 specs más repetidos en el año indicado
@app.get('/specs/')
def specs(año: int):
    filtered_data_steam = data_steam[data_steam['release_date'].dt.year == año]
    exploded_specs_data_steam = filtered_data_steam.explode('specs')
    top_specs = exploded_specs_data_steam['specs'].value_counts().nlargest(5).index.tolist()
    return top_specs

# Retorna cantidad de juegos lanzados con early acces en el año indicado
@app.get('/earlyacces/')
def earlyacces(año: int):
    filtered_data_steam = data_steam[data_steam['release_date'].dt.year == año]
    count_early_access = len(filtered_data_steam[filtered_data_steam['early_access'] == True])
    return count_early_access

# Retorna lista con registros categorizados con un "sentiment" específico, en el año indicado
@app.get('/sentiment/')
def sentiment(año: int):
    filtered_data_steam = data_steam[data_steam['release_date'].dt.year == año]
    sentiment_counts = filtered_data_steam['sentiment'].value_counts().to_dict()
    return sentiment_counts

# Retorna los 5 juegos con mayor metascore en el año indicado
@app.get('/metascore/')
def metascore(año: int):
    filtered_data_steam = data_steam[data_steam['release_date'].dt.year == año]
    top_metascore_games = filtered_data_steam.nlargest(5, 'metascore')[['app_name', 'metascore']].set_index('app_name').to_dict()['metascore']
    return top_metascore_games

steam_unnested = data_steam.explode('genres')

# Convertir 'release_date' a año
steam_unnested['release_year'] = steam_unnested['release_date'].dt.year

# Convertir 'genres' a números (usando one-hot encoding)
steam_dummies = pd.get_dummies(steam_unnested, columns=['genres'], prefix='', prefix_sep='')

# Dividir en entrenamiento y prueba
X = steam_unnested[['release_year', 'metascore'] + list(steam_dummies.columns[steam_dummies.columns.str.contains('genres')])]
y = steam_unnested['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear características polinomiales
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Entrenar modelo
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Evaluar modelo
y_pred = model.predict(X_test_poly)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# GET para obtener la predicción de precio y RMSE
@app.get("/prediccion/")
async def get_prediccion(
    genero: str = Query(
        ...,  # Esto significa que el parámetro es requerido
        description="El género del juego debe ser uno de los siguientes: " + ', '.join(steam_unnested.columns[steam_unnested.columns.str.contains('genres')]),
    ),
    año: int = Query(
        ...,  # Esto significa que el parámetro es requerido
        description="El año de lanzamiento del juego.",
    ),
    metascore: int = Query(
        ...,  # Esto significa que el parámetro es requerido
        description="El Metascore del juego.",
    )
):
    # Convertir 'genero' a números (usando one-hot encoding)
    genres = list(steam_unnested.columns[steam_unnested.columns.str.contains('genres')])
    if genero not in genres:
        raise HTTPException(status_code=400, detail="Género no válido. Por favor use un género de la lista de géneros disponibles.")
    genre_data = [1 if genero == genre else 0 for genre in genres]
    data = np.array([año, metascore] + genre_data).reshape(1, -1)
    
    # Aplicar la transformación polinomial
    data_poly = poly.transform(data)

    price = model.predict(data_poly)[0]
    return {'price': price, 'rmse': rmse}
