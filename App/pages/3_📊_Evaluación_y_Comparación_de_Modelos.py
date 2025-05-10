import streamlit as st
import utils 
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Evaluación y Comparación de modelos", page_icon="📊",layout="wide")


st.markdown(
    f"<h1 style='text-align: center;'>Evaluación y comparación de modelos</h1>",
    unsafe_allow_html=True
)


# Cargar DataFrame precalculado
df_errores = utils.cargar_errores_modelos()

# Selector de sector (con "Todos")
sectores = ["Todos"] + sorted(df_errores["SECTOR"].unique())
sector_sel = st.selectbox("Selecciona sector:", sectores)

# Selector de horizonte
horizonte_tipo = st.radio("Tipo de horizonte temporal:", ["Corto (1-2 años)", "Largo (3-5 años)", "Manual"])
if horizonte_tipo == "Manual":
    horizonte_valores = [st.slider("Selecciona horizonte:", 1, 5, 3)]
elif "Corto" in horizonte_tipo:
    horizonte_valores = [1, 2]
else:
    horizonte_valores = [3, 4, 5]

# ---- FILTRADO DE DATOS ----
df_filtrado = df_errores[df_errores["HORIZONTE"].isin(horizonte_valores)]

# Excluir "España" si se selecciona "Todos"
if sector_sel == "Todos":
    df_filtrado = df_filtrado[df_filtrado["SECTOR"] != "España"]
else:
    df_filtrado = df_filtrado[df_filtrado["SECTOR"] == sector_sel]

# ---- VISUALIZACIÓN ----

# 🟣 Radar chart (por modelo)
if sector_sel != "Todos":
    st.markdown("### Métricas por modelo")

    df_radar = df_filtrado.copy()

    # Si no es manual, agregamos por media para ese horizonte múltiple
    if horizonte_tipo != "Manual":
        df_radar = df_radar.groupby("MODELO")[["MAE", "RMSE", "MAPE"]].mean()
    else:
        df_radar.set_index("MODELO", inplace=True)
        df_radar = df_radar[["MAE", "RMSE", "MAPE"]]

    # Normalización con StandardScaler
    scaler = StandardScaler()
    df_norm = pd.DataFrame(
        scaler.fit_transform(df_radar),
        columns=df_radar.columns,
        index=df_radar.index
    )

    fig_radar = go.Figure()
    for modelo in df_norm.index:
        valores = df_norm.loc[modelo].values
        categorias = df_norm.columns

        color = utils.coloresModelos.get(modelo.upper(), "#888888")

        fig_radar.add_trace(go.Scatterpolar(
            r=list(valores) + [valores[0]],
            theta=list(categorias) + [categorias[0]],
            fill='toself',
            name=modelo,
            line=dict(color=color),
            fillcolor=utils.hex_to_rgba(color, alpha=0.2)
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-2, 2])), showlegend=True)
    st.plotly_chart(fig_radar)

# 🔥 Heatmap por métrica y modelo (solo si "Todos")
if sector_sel == "Todos":
    df_filtrado = df_filtrado[df_filtrado["SECTOR"] != "ESPAÑA"]
    min_val = df_filtrado[["MAE", "RMSE"]].min().min()
    max_val = df_filtrado[["MAE", "RMSE"]].max().max()
    st.markdown("### Mapa de calor por medida")
    metrica_seleccionada = st.selectbox("Selecciona la medida para el mapa de calor:", ["MAE", "RMSE", "MAPE"])
    st.subheader(metrica_seleccionada)

    heat_df = df_filtrado.pivot_table(index="SECTOR", columns="MODELO", values=metrica_seleccionada, aggfunc="mean")
    fig_heat = px.imshow(
        heat_df,
        color_continuous_scale=px.colors.sequential.Teal,  # O Inferno
        aspect="auto",
        zmin=min_val if metrica_seleccionada in ["MAE", "RMSE"] else None,
        zmax=max_val if metrica_seleccionada in ["MAE", "RMSE"] else None
    )

    st.plotly_chart(fig_heat)

# 📊 Gráficos de barras en dos columnas
st.markdown("### Comparación en gráficos de barras")
col1, col2 = st.columns(2)

df_melt = df_filtrado.melt(
    id_vars=["SECTOR", "MODELO", "HORIZONTE"],
    value_vars=["MAE", "RMSE", "MAPE"],
    var_name="Métrica", value_name="Valor"
)
colores_modelos = utils.coloresModelos  # Ya definidos por ti

# Agregar columna con color y patrón por modelo y métrica
df_melt["MODELO_ESTD"] = df_melt["MODELO"].apply(modelo.upper())
df_melt["Color"] = df_melt["MODELO_ESTD"].map(utils.coloresModelos)
df_melt["Pattern"] = df_melt["Métrica"].apply(lambda x: "/" if x == "RMSE" else "")

# 📊 MAE y RMSE
with col1:
    df_m1 = df_melt[df_melt["Métrica"].isin(["MAE", "RMSE"])]
    fig_bar_1 = px.bar(
        df_m1,
        x="SECTOR" if sector_sel == "Todos" else "MODELO",
        y="Valor",
        color="MODELO_ESTD",
        barmode="group",
        pattern_shape="Métrica",
        pattern_shape_map={"RMSE": "/", "MAE": ""},
        color_discrete_map=utils.coloresModelos,
        title="MAE y RMSE"
    )
    #fig_bar_1.update_traces(width=0.3)
    st.plotly_chart(fig_bar_1)


# 📊 MAPE (solo color plano)
with col2:
    df_m2 = df_melt[df_melt["Métrica"] == "MAPE"]
    fig_bar_2 = px.bar(
        df_m2,
        x="SECTOR" if sector_sel == "Todos" else "MODELO",
        y="Valor",
        color="MODELO_ESTD",
        barmode="group",
        color_discrete_map=utils.coloresModelos,
        title="MAPE"
    ) 
    #fig_bar_2.update_traces(width=0.3)
    st.plotly_chart(fig_bar_2)

# 📋 Tabla con los errores
st.markdown("### Tabla de métricas")
st.dataframe(df_filtrado.set_index(["SECTOR", "MODELO"]).round(2))

