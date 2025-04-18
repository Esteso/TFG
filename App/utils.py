import pandas as pd
import streamlit as st
import itertools
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import  mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.cross_decomposition import PLSRegression       
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import os

def load_data(): 
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data', 'datosPanel.csv')
    data_path = os.path.normpath(os.path.join(base_path, 'data', 'datosPanel.csv'))
    df=pd.read_csv(data_path, delimiter=";")
    df = df.replace(',', '.', regex=True)
    columnas_numericas = ["SOLVENCIA", "LIQUIDEZ", "ACTIVO FIJO", "ROE (%)", "PRODUCTIVIDAD HUMANA","RATIO K/L", "ENDEUDAMIENTO (%)"]
    for col in columnas_numericas: 
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convierte a float, NaN si falla
    return df

def load_inflation_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data', 'datosCorregidos.csv')
    data_path = os.path.normpath(os.path.join(base_path, 'data', 'datosPanel.csv'))
    df_inflacion = pd.read_csv(data_path, delimiter=";")
    df_inflacion = df_inflacion.replace(',', '.', regex=True)

    columnas_numericas = ["VALOR AÑADIDO*", "GASTOS PERSONAL*", "ACTIVO TOTAL*", "ACTIVO FIJO*", "RATIO K/L*"]
    for col in columnas_numericas:
        df_inflacion[col] = pd.to_numeric(df_inflacion[col], errors='coerce')  # Convertir a float

    return df_inflacion.set_index(["SECTOR","AÑO"])

def load_allData():
    return pd.concat([load_data().set_index(["SECTOR","AÑO"]), load_inflation_data()], axis=1).reset_index(drop=False)

def cargar_errores_modelos():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, 'data', 'datosCorregidos.csv')
    data_path = os.path.normpath(os.path.join(base_path, 'data', 'errores_modelos.csv'))
    return pd.read_csv(data_path)

def seleccionar_mejor_arima(serie, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3),horizonte=2):
    mejor_rmse = float("inf")
    mejor_orden = None
    train=serie[:-horizonte]
    test=serie[-horizonte:]
    # Generamos todas las combinaciones posibles de (p, d, q)
    for p, d, q in itertools.product(range(p_range[0], p_range[1] + 1),
                                    range(d_range[0], d_range[1] + 1),
                                    range(q_range[0], q_range[1] + 1)):
        try:
            # Ajustamos el modelo ARIMA
            modelo = ARIMA(train, order=(p, d, q)).fit()
            
            # Hacemos la predicción en el mismo rango de la serie
            predicciones = modelo.forecast(steps=horizonte)

            # Calculamos RMSE
            rmse = np.sqrt(mean_squared_error(test, predicciones))
            # Si encontramos un mejor modelo, lo guardamos
            if rmse < mejor_rmse:
                mejor_rmse = rmse
                mejor_orden = (p, d, q)
        
        except:
            continue  # Ignorar combinaciones que no converjan
    
    return mejor_orden

def opciones_ordenArima():
    combinaciones=list(itertools.product([0,1,2,3],[0,1,2],[0,1,2,3]))
    return combinaciones

miscolores = {
    "TURISMO": "#A833FF",
    "TRANSPORTE": "#E63946",
    "TECN. INFORMACIÓN": "#00BFFF",
    "CONSTRUCCIÓN": "#FF8C00",
    "AGROALIMENTARIA": "#00C853",
    "ESPAÑA": "#000000"
}

coloresModelos={
    "ARIMA":"#EF2C1B",
    "SUAVIZADO EXPONENCIAL":"#F0BE09",
    "REGRESIÓN PLS":"#2D1FF0",
    "REGRESIÓN + PCA":"#1FF07C",

}

modelo_aliases = {
    "Regresión múltiple + PCA": "REGRESIÓN + PCA",
    "PCA + REGRESIÓN": "REGRESIÓN + PCA",
    "REGRESIÓN + PCA": "REGRESIÓN + PCA",
    "Regresión PLS": "REGRESIÓN PLS",
    "REGRESIÓN PLS": "REGRESIÓN PLS",
    "Suavizado Exponencial": "SUAVIZADO EXPONENCIAL",
    "SUAVIZADO EXPONENCIAL": "SUAVIZADO EXPONENCIAL",
    "ARIMA": "ARIMA"
}

def modelo_estandar(nombre):
    return modelo_aliases.get(nombre, nombre.upper())

def hex_to_rgba(hex_color, alpha=0.4):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def predecir_modelo(df, sector, horizonte, modelo, pred_futura=False, orden_manual=None):
    df_sector = df[df["SECTOR"] == sector].sort_values("AÑO")

    if modelo == "ARIMA":
        serie = df_sector.set_index("AÑO")["VALOR AÑADIDO (MIL €)"].sort_index()

        if pred_futura:
            X_train = serie
            vReales = None
            import pmdarima as pm
            orden = pm.auto_arima(X_train,seasonal=False,stepwise=True,suppress_warnings=True,trace=False).order
        else:
            X_train = serie[:-horizonte]
            vReales = serie[-horizonte:]
            if orden_manual == "ORDEN AUTO ARIMA":
                import pmdarima as pm
                orden = pm.auto_arima(X_train,seasonal=False,stepwise=True,suppress_warnings=True,trace=False).order
            elif orden_manual == "MEJOR ORDEN ENCONTRADO":
                orden = seleccionar_mejor_arima(serie, horizonte=horizonte)
            elif orden_manual!=None:
                orden = orden_manual
            else:         
                orden = seleccionar_mejor_arima(serie, horizonte=horizonte)
        modelo_fit = ARIMA(X_train, order=orden).fit()
        predicciones = modelo_fit.forecast(steps=horizonte)
        return predicciones, vReales

    elif modelo == "SUAVIZADO EXPONENCIAL":
        Y = df_sector["VALOR AÑADIDO (MIL €)"].values

        if pred_futura:
            X_train = Y
            vReales = None
        else:
            X_train = Y[:-horizonte]
            vReales = Y[-horizonte:]

        modelo = ExponentialSmoothing(X_train, trend="additive", seasonal=None)
        fit = modelo.fit()
        predicciones = fit.forecast(horizonte)
        return predicciones, vReales

    elif modelo == "REGRESIÓN PLS":
        X_full = df_sector.drop(columns=['AÑO', 'VALOR AÑADIDO (MIL €)', 'SECTOR'])
        y_full = df_sector['VALOR AÑADIDO (MIL €)']

        if pred_futura:
            X_train = X_full
            y_train = y_full
            # Duplicamos el último registro para generar inputs simulados
            X_test = pd.concat([X_full.tail(1)] * horizonte, ignore_index=True)
            vReales = None
        else:
            train_year_cutoff = 2023 - horizonte
            df_train = df_sector[df_sector["AÑO"] <= train_year_cutoff]
            df_test = df_sector[df_sector["AÑO"] > train_year_cutoff]

            X_train = df_train.drop(columns=['AÑO', 'VALOR AÑADIDO (MIL €)', 'SECTOR'])
            y_train = df_train['VALOR AÑADIDO (MIL €)']
            X_test = df_test.drop(columns=['AÑO', 'VALOR AÑADIDO (MIL €)', 'SECTOR'])
            vReales = df_test['VALOR AÑADIDO (MIL €)']

        pls = PLSRegression(n_components=2)
        pls.fit(X_train, y_train)
        predicciones = pls.predict(X_test).flatten()
        return predicciones, vReales

    elif "REGRESIÓN + PCA" == modelo:
        sector_data = df_sector.drop(columns=["SECTOR"])
        X = sector_data.drop(columns=["VALOR AÑADIDO (MIL €)", "AÑO"]).reset_index(drop=True)
        Y = sector_data["VALOR AÑADIDO (MIL €)"]

        if pred_futura:
            X_train = X
            y_train = Y
            X_test = pd.concat([X.tail(1)] * horizonte, ignore_index=True)
            vReales = None
        else:
            X_train = X[:-horizonte]
            X_test = X[-horizonte:]
            y_train = Y[:-horizonte]
            vReales = Y[-horizonte:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=0.9)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        reg = LinearRegression()
        reg.fit(X_train_pca, y_train)
        predicciones = reg.predict(X_test_pca)
        return predicciones, vReales

    else:
        raise ValueError(f"Modelo '{modelo}' no implementado.")