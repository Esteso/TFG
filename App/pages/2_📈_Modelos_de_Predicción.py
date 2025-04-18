import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import pandas as pd
import utils

st.set_page_config(page_title="Modelos de Predicción", page_icon="📈",layout="wide")


st.markdown(
    f"<h1 style='text-align: center;'>Modelos de predicción</h1>",
    unsafe_allow_html=True
)


df = utils.load_data()

col1, col2 = st.columns(2)

with col1:
    modelo_seleccionado = st.selectbox(
        "Selecciona un modelo de predicción:",
        ["Regresión múltiple + PCA", "ARIMA", "Suavizado Exponencial", "Regresión PLS", "Todos"]
    )

with col2:
    sector_a_predecir = st.selectbox(
        "Selecciona el sector a predecir:",
        df["SECTOR"].unique()
    )
# Para un único modelo

st.markdown("#### Opciones de predicción")
pred_futura = st.checkbox("Hacer predicción a futuro (sin comparar con datos reales)")

if pred_futura:
    horizonte = st.slider("¿Cuántos años quieres predecir?", min_value=1, max_value=3, value=1)
    orden=None
else:
    if modelo_seleccionado=="ARIMA":
        combis=utils.opciones_ordenArima()
        orden=st.selectbox("Seleccione orden para ARIMA:",[f"ORDEN AUTO ARIMA",f"MEJOR ORDEN ENCONTRADO"]+combis)
    horizonte = st.slider("Selecciona el horizonte temporal:", min_value=1, max_value=5, value=2)

# Serie histórica
df_sector = df[df["SECTOR"] == sector_a_predecir].sort_values("AÑO")
VAtoshow = df_sector["VALOR AÑADIDO (MIL €)"].values
fig = go.Figure()

# Línea de la serie histórica
fig.add_trace(go.Scatter(
    x=[i for i in range(2013, 2024)],
    y=VAtoshow,
    mode='lines+markers',
    name="Serie histórica",
    line=dict(color="grey")
))
if modelo_seleccionado == "Todos":
    modelos_ya_mostrados = set()
    for modelo, color in utils.coloresModelos.items():
        modelo_estandarizado = utils.modelo_estandar(modelo)
        if modelo_estandarizado in modelos_ya_mostrados:
            continue  # ya lo hemos graficado
        modelos_ya_mostrados.add(modelo_estandarizado)
        predicciones, _ = utils.predecir_modelo(df, sector_a_predecir, horizonte, modelo_estandarizado, pred_futura=pred_futura)
        x_pred = [2023] + list(range(2024, 2024 + horizonte)) if pred_futura else list(range(2023 - horizonte, 2024))
        y_pred = [list(VAtoshow)[-1]] + list(predicciones) if pred_futura else [list(VAtoshow)[-horizonte-1]] + list(predicciones)
        
        fig.add_trace(go.Scatter(
            x=x_pred,
            y=y_pred,
            mode='lines+markers',
            name=modelo_estandarizado,
            line=dict(color=color, dash="dash")
        ))
    fig.update_layout(
    title=f"Predicción de {horizonte} año(s) para {sector_a_predecir} usando TODOS los modelos",
    xaxis_title="Año",
    yaxis_title="Valor añadido (mil €)",
    template="plotly_white"
    )
    st.plotly_chart(fig)
    st.info("Para comparar los errores de predicción, ve a la sección 'Comparación y evaluación de modelos'.")
else:
    modelo_key = utils.modelo_estandar(modelo_seleccionado)
    if modelo_seleccionado=="ARIMA":
        predicciones, vReales = utils.predecir_modelo(df, sector_a_predecir, horizonte, modelo_key, pred_futura=pred_futura,orden_manual=orden)
    else:
        predicciones, vReales = utils.predecir_modelo(df, sector_a_predecir, horizonte, modelo_key, pred_futura=pred_futura)

    colorModelo = utils.coloresModelos[modelo_key]
    x_pred = [2023] + list(range(2024, 2024 + horizonte)) if pred_futura else list(range(2023 - horizonte, 2024))
    y_pred = [list(VAtoshow)[-1]] + list(predicciones) if pred_futura else [list(VAtoshow)[-horizonte-1]] + list(predicciones)

    fig.add_trace(go.Scatter(
        x=x_pred,
        y=y_pred,
        mode='lines+markers',
        name="Predicción",
        line=dict(color=colorModelo, dash="dash")
    ))

    fig.update_layout(
        title=f"Predicción de {horizonte} año(s) para {sector_a_predecir} usando {modelo_seleccionado}",
        xaxis_title="Año",
        yaxis_title="Valor añadido (mil €)",
        legend=dict(x=0, y=1),
        template="plotly_white"
    )

    st.plotly_chart(fig)

    if not pred_futura:
        mae = mean_absolute_error(vReales, predicciones)
        mape = mean_absolute_percentage_error(vReales, predicciones)
        rmse = mean_squared_error(vReales, predicciones, squared=False)

        errores_df = pd.DataFrame({
            "Métricas de error": ['MAE', 'MAPE', 'RMSE'],
            "Valor": [mae, mape, rmse]
        }).set_index("Métricas de error").round(3)

        errores_df["Valor"] = errores_df["Valor"].apply(lambda x: f"{x:.3f}")
        st.table(errores_df)
    else:
        st.info("Predicción realizada a futuro. No se muestran errores por falta de valores reales.")
