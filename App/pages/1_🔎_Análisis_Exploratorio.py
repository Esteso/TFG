import streamlit as st
import utils 
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Análisis Exploratorio", page_icon="🔎",layout="wide")


st.markdown(
    f"<h1 style='text-align: center;'>Análisis exploratorio</h1>",
    unsafe_allow_html=True
)

df=utils.load_allData()
st.dataframe(df.set_index(["SECTOR","AÑO"]))
col_espacio, col_letra = st.columns([4, 1])
with col_letra:
    st.write(f"\* variables ajustadas a la inflación")

with col_espacio:
    inflacion=st.checkbox("Comparar datos nominales vs reales corregidos por inflación")

#st.dataframe(df.set_index(["SECTOR","AÑO"]))
col1, col2 = st.columns(2)

with col1:
    metricas=["VALOR AÑADIDO","GASTOS PERSONAL","ACTIVO TOTAL","RATIO K/L","GASTOS PERSONAL/EMPLEADO","PRODUCTIVIDAD"] if inflacion else list(df.columns)[2:-5]
    dfl="VALOR AÑADIDO" if inflacion else "VALOR AÑADIDO (MIL €)"
    metricas_seleccionadas = st.multiselect(
    "Selecciona hasta 4 métricas para visualizar:", metricas, default=dfl,
    max_selections=4
    )
    
with col2: 
    if inflacion: 
        sector = st.selectbox("Selecciona un sector:", df["SECTOR"].unique())
        df_filtrado = df[df["SECTOR"]==sector]

    else: 
        sectores_seleccionados = st.multiselect("Selecciona sectores:", df["SECTOR"].unique(), default=[df["SECTOR"].unique()[3]])
        df_filtrado = df[df["SECTOR"].isin(sectores_seleccionados)]

# Creación de gráficos
figs = []
if inflacion: 
    for metrica in metricas_seleccionadas:
        if "GASTOS" in metrica:
            serie1=df_filtrado["GASTOS PERSONAL (MIL €) "]
            serie2=df_filtrado["GASTOS PERSONAL*"]
            if "/" in metrica: 
                serie1=serie1/df_filtrado["EMPLEADOS"]
                serie2=serie2/df_filtrado["EMPLEADOS"]
        elif "K/L" in metrica: 
            serie1=df_filtrado[metrica]
            serie2=df_filtrado[metrica+"*"]
        elif "PRODUCTIVIDAD"==metrica: 
            serie1=df_filtrado["VALOR AÑADIDO (MIL €)"]/df_filtrado["EMPLEADOS"]
            serie2=df_filtrado["VALOR AÑADIDO*"]/df_filtrado["EMPLEADOS"]
        else:
            serie1=df_filtrado[metrica+" (MIL €)"]
            serie2=df_filtrado[metrica+"*"] 
        fig = go.Figure()
        fig.add_trace(go.Scatter(
                x=df_filtrado["AÑO"], 
                y=serie1, 
                mode='lines', 
                name=f"{sector} - {metrica} nominal", 
                line=dict(color=utils.miscolores[sector], dash="solid")  # Línea continua
            ))
        fig.add_trace(go.Scatter(
            x=df_filtrado["AÑO"], 
            y=serie2, 
            mode='lines', 
            name=f"{sector} - {metrica} real", 
            line=dict(color=utils.miscolores[sector], dash="dash")  # Línea discontinua
        ))

        fig.update_layout(title=f"Comparación de {metrica} con su corrección por la inflación",
                        xaxis_title="AÑO",
                        legend_title="Sector y Variable",legend=dict(
                        orientation="h",  # Orientación horizontal
                        yanchor="bottom",  # Ancla de la leyenda en la parte inferior
                        y=-0.3,  # Desplazamiento de la leyenda hacia abajo
                        xanchor="center",  # Centra la leyenda
                        x=0.5  ))
        figs.append(fig)
else:
    for metrica in metricas_seleccionadas:
        log_y = True if metrica not in ["ROE (%)", "ENDEUDAMIENTO (%)", "SOLVENCIA", "LIQUIDEZ","RATIO K/L"] and "ESPAÑA" in sectores_seleccionados and len(sectores_seleccionados)>1 else False

        fig = px.line(
            df_filtrado, x="AÑO", y=metrica, color="SECTOR",
            title=f"Evolución de {metrica}",
            color_discrete_map=utils.miscolores,
            log_y=log_y
        )
        figs.append(fig)

# Mostrar gráficos según el número de métricas seleccionadas
if len(figs) == 1:
    st.plotly_chart(figs[0])
elif len(figs) == 2:
    st.plotly_chart(figs[0])
    st.plotly_chart(figs[1])
elif len(figs) == 3:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(figs[0])
    with col2:
        st.plotly_chart(figs[1])
    st.plotly_chart(figs[2])
elif len(figs) == 4:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(figs[0])
    with col2:
        st.plotly_chart(figs[1])
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(figs[2])
    with col4:
        st.plotly_chart(figs[3])
