import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Várzea Inteligente",
    page_icon="🌊",
    layout="wide"
)

df = pd.read_csv("rio_negro.csv")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data")

nivel = df["nivel"].iloc[-1]
inicial = df["nivel"].iloc[0]
variacao = (nivel - inicial) * 100

st.title("🌊 VÁRZEA INTELIGENTE")
st.write("Monitoramento do Rio Negro — Manaus")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Nível registrado",
        f"{nivel:.2f} m"
    )

with col2:
    st.metric(
        "Variação no período",
        f"{variacao:.0f} cm"
    )

if variacao < 0:
    st.info("📉 Rio em descida")
elif variacao > 0:
    st.info("📈 Rio em subida")
else:
    st.info("➖ Nível estável")

st.subheader("📊 Histórico do Rio Negro")

st.line_chart(
    df.set_index("data")["nivel"]
)

st.caption(
    "Último registro: " +
    df["data"].max().strftime("%d/%m/%Y")
)

st.warning(
    "Dados históricos. Este sistema ainda "
    "não realiza previsões com IA e não "
    "substitui alertas oficiais."
)

st.caption("Fonte: Porto de Manaus")
