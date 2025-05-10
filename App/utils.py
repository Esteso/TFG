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
from sklearn.linear_model import LinearRegression,LassoCV
import pmdarima as pm
import os
import ast
import statsmodels.api as sm



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
    data_path = os.path.normpath(os.path.join(base_path, 'data', 'datosCorregidos.csv'))
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

def seleccionar_mejor_arima(sector, horizonte):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(base_dir, "..", "data", "df_ordenesArima.csv")
    df_ordenes = pd.read_csv(ruta_csv)
    df_ordenes["orden"] = df_ordenes["orden"].apply(ast.literal_eval)
    fila = df_ordenes[
        (df_ordenes["sector"] == sector) & (df_ordenes["horizonte"] == horizonte)
    ]
    if not fila.empty:
        return fila.iloc[0]["orden"]
    else:
        raise ValueError(f"No se encontró orden para sector={sector} y horizonte={horizonte}")

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

colores_modelos = {
    "ARIMA": "#EF2C1B",
    "SUAVIZADO EXPONENCIAL": "#F0BE09",
    "REGRESIÓN KOYCK": "#2D1FF0",
    "REGRESIÓN LASSO": "#1FF07C"
}

coloresModelos = {
    "ARIMA": "#EF2C1B",
    "SUAVIZADO EXPONENCIAL": "#F0BE09",
    "REGRESIÓN KOYCK": "#2D1FF0",
    "REGRESIÓN LASSO": "#1FF07C"
}

def hay_tendencia(serie, umbral_porcentual=10):
    primer_valor = serie[0]
    ultimo_valor = serie[-1]
    cambio_relativo = (ultimo_valor - primer_valor) / abs(primer_valor) * 100
    return abs(cambio_relativo) >= umbral_porcentual

def hex_to_rgba(hex_color, alpha=0.4):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def predecir_modelo(df, sector, horizonte, modelo, pred_futura=False, orden_manual=None):
    if modelo == "ARIMA":
        df_sector = df[df["SECTOR"] == sector].sort_values("AÑO")
        serie = df_sector.set_index("AÑO")["VALOR AÑADIDO (MIL €)"].sort_index()
        
        if pred_futura:
            X_train = serie
            vReales = None
            forzar_d = hay_tendencia(list(X_train))
            orden = pm.auto_arima(X_train,seasonal=False,stepwise=False,suppress_warnings=True,trace=False,error_action="ignore",test="kpss",start_p=1,start_q=1,max_p=3,max_q=3,d=1 if forzar_d else None).order
            p, d, q = orden
            if p == 0 and q == 0:
                orden=(1,1,1)        
        else:
            X_train = serie[:-horizonte]
            vReales = serie[-horizonte:]
            if orden_manual == "ORDEN AUTO ARIMA":
                forzar_d = hay_tendencia(list(X_train))
                orden = pm.auto_arima(X_train,seasonal=False,stepwise=False,suppress_warnings=True,trace=False,error_action="ignore",test="kpss",start_p=1,start_q=1,max_p=3,max_q=3,d=1 if forzar_d else None).order
                p, d, q = orden
                if p == 0 and q == 0:
                    orden=(1,1,1)
            elif orden_manual == "MEJOR ORDEN ENCONTRADO":
                orden = seleccionar_mejor_arima(serie, horizonte=horizonte)
            elif orden_manual!=None:
                orden = orden_manual
            else:   
                forzar_d = hay_tendencia(list(X_train))
                orden = pm.auto_arima(X_train,seasonal=False,stepwise=False,suppress_warnings=True,trace=False,error_action="ignore",test="kpss",start_p=1,start_q=1,max_p=3,max_q=3,d=1 if forzar_d else None).order
                p, d, q = orden
                if p == 0 and q == 0:
                    orden=(1,1,1)
        modelo_fit = ARIMA(X_train, order=orden).fit()
        predicciones = modelo_fit.forecast(steps=horizonte)
        return predicciones, vReales

    elif modelo == "SUAVIZADO EXPONENCIAL":
        df_sector = df[df["SECTOR"] == sector].sort_values("AÑO")
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

    elif modelo == "REGRESIÓN KOYCK":
        df=df.reset_index(drop=False)
        df_sector = df[df['SECTOR'] == sector].sort_values('AÑO').reset_index(drop=True)
        df_sector['VA_lag1'] = df_sector['VALOR AÑADIDO (MIL €)'].shift(1)
        df_sector = df_sector.dropna().reset_index(drop=True)
        if pred_futura:
            df_entrenamiento = df_sector.copy()
        else:
            df_entrenamiento = df_sector.iloc[:-horizonte].copy()
        Y_train = df_entrenamiento['VALOR AÑADIDO (MIL €)']
        X_train = df_entrenamiento[['VA_lag1']]
        X_train = sm.add_constant(X_train)
        modelo = sm.OLS(Y_train, X_train).fit()
        df_pred = df_entrenamiento.copy() if not pred_futura else df_sector.copy()
        current_year = df_pred['AÑO'].max()
        predicciones = []
        años_predichos = []
        alpha = modelo.params['const']
        beta = modelo.params['VA_lag1']
        current_value = df_pred['VALOR AÑADIDO (MIL €)'].iloc[-1]
        for i in range(horizonte):
            next_year = current_year + 1
            next_value = alpha + beta * current_value
            predicciones.append(next_value)
            años_predichos.append(next_year)
            current_value = next_value
            current_year = next_year
        Y = df_sector["VALOR AÑADIDO (MIL €)"]
        vReales=Y[-horizonte:]
        return predicciones, vReales

    elif modelo=="REGRESIÓN LASSO":
        lags=3
        target_col = 'VALOR AÑADIDO (MIL €)'
        df = df.reset_index(drop=False)
        df_sector = df[df['SECTOR'] == sector].sort_values('AÑO').copy()
        df_sector = df_sector[df_sector[target_col] > 0].reset_index(drop=True)
        Y = df_sector[target_col]
        vReales = Y[-horizonte:]
        for lag in range(1, lags + 1):
            df_sector[f'{target_col}_lag{lag}'] = df_sector[target_col].shift(lag)
        df_sector = df_sector.dropna().reset_index(drop=True)
        if pred_futura:
            df_entrenamiento = df_sector.copy()
        else:
            df_entrenamiento = df_sector.iloc[:-horizonte].copy()
        y_train = df_entrenamiento[target_col].values
        X_train = df_entrenamiento[[f'{target_col}_lag{lag}' for lag in range(1, lags + 1)]].values
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        if horizonte==5:
            cv_=3
        elif horizonte==4:
            cv_=4
        else: 
            cv_=5
        modelo = LassoCV(cv=cv_, random_state=0)
        modelo.fit(X_train_scaled, y_train)
        df_pred = df_entrenamiento.copy()
        current_year = df_entrenamiento['AÑO'].max()
        predicciones = []
        años_predichos = []
        for _ in range(horizonte):
            next_year = current_year + 1
            pred_input = []
            for lag in range(1, lags + 1):
                año_lag = next_year - lag

                fila_lag = df_pred[df_pred['AÑO'] == año_lag]
                if fila_lag.empty:
                    raise ValueError(f"No hay valor para el año {año_lag}.")
                val = fila_lag[target_col].values[0]
                pred_input.append(val)
            X_pred = np.array(pred_input).reshape(1, -1)
            X_pred_scaled = scaler.transform(X_pred)
            y_pred = modelo.predict(X_pred_scaled)[0]

            predicciones.append(y_pred)
            años_predichos.append(next_year)
            new_row = {
            'AÑO': next_year,
            target_col: y_pred,
            'SECTOR': sector
            }
            for lag in range(1, lags + 1):
                new_row[f'{target_col}_lag{lag}'] = df_pred[df_pred['AÑO'] == (next_year - lag)][target_col].values[0]

            df_pred = pd.concat([df_pred, pd.DataFrame([new_row])], ignore_index=True)
            current_year = next_year
        return predicciones, vReales

    else:
        raise ValueError(f"Modelo '{modelo}' no implementado.")