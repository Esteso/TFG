import streamlit as st

# Estilo general

st.set_page_config(page_title="TFG - INICIO", page_icon="🏠",layout="wide")
#0_🏠_INICIO.py

st.markdown("""
<div style='background-color: #1565c0; padding: 30px; border-radius: 10px; text-align: center; color: white'>
    <h1 style='font-size: 3em;'>📊 Análisis y Predicción de Sectores Clave</h1>
    <p style='font-size: 1.3em;'>Herramienta interactiva para visualizar el pasado y futuro de la economía española</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")
st.markdown("")

# Breve descripción
st.markdown("""

Esta aplicación permite explorar la evolución y realizar predicciones sobre indicadores económicos clave de varios sectores de la economía española.

Las funcionalidades principales se dividen en tres secciones:

- 🔍 **Análisis Exploratorio**  
  Visualiza y compara indicadores históricos entre sectores clave, como turismo, agroalimentario, tecnología, construcción o transporte.

- 📈 **Modelos de Predicción**  
  Aplica distintos modelos de predicción (ARIMA, Suavizado Exponencial, Regresión de Koyck, Regresión LASSO con rezagos) para estimar el valor añadido futuro.

- 📊 **Evaluación y Comparación de Modelos**  
  Compara el rendimiento de los modelos mediante métricas como MAE, RMSE y MAPE, con visualizaciones en radar, heatmaps y barras.

---

👨‍🎓 **Autor:** Jose Manuel Esteso Morcillo  
📚 **Grado en Ciencia de Datos** – ETSINF – UPV  
📅 **Curso 2024/2025**
""")

st.markdown("""
<div style="background-color: #e3f2fd; padding: 20px 30px; border-radius: 10px; border-left: 6px solid #2196F3;">
    <h3 style="color: #1565c0;">🧭 Navegación</h3>
    <ul style="font-size: 17px; color: #0d47a1;">
        <li><a href="Análisis_Exploratorio" style="color: #1976d2; text-decoration: none;"><strong>🔍 Ir a Análisis Exploratorio</strong></a> – para visualizar los datos históricos.</li>
        <li><a href="/Modelos_de_Predicción" style="color: #1976d2; text-decoration: none;"><strong>📈 Ir a Modelos de Predicción</strong></a> – para probar diferentes modelos ante distintos escenarios.</li>
        <li><a href="/Evaluación_y_Comparación_de_Modelos" style="color: #1976d2; text-decoration: none;"><strong>📊 Ir a Evaluación de Modelos</strong></a> – para analizar los resultados de los modelos.</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# Puedes agregar una imagen, logo o gráfico aquí si quieres
# st.image("logo_upv.png", width=200)  # si tienes un logo local
