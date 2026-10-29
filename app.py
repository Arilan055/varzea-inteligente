import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Várzea Inteligente",
    page_icon="🌊",
    layout="wide"
)

# Carregar dados
df = pd.read_csv("rio_negro.csv")
df["data"] = pd.to_datetime(df["data"])
df = df.sort_values("data").reset_index(drop=True)

# Informações históricas
nivel = df["nivel"].iloc[-1]
inicial = df["nivel"].iloc[0]
variacao = (nivel - inicial) * 100

st.title("🌊 VÁRZEA INTELIGENTE")
st.write("Monitoramento do Rio Negro — Manaus")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.metric("Último nível observado", f"{nivel:.2f} m")

with col2:
    st.metric("Variação no período", f"{variacao:.0f} cm")

if variacao < 0:
    st.info("📉 Rio em descida")
elif variacao > 0:
    st.info("📈 Rio em subida")
else:
    st.info("➖ Nível estável")

st.subheader("📊 Histórico do Rio Negro")
st.line_chart(df.set_index("data")["nivel"])

st.caption(
    "Última medição da base: " +
    df["data"].max().strftime("%d/%m/%Y")
)

# Preparar dados da IA
dados = df.copy()
dados["dia_1"] = dados["nivel"].shift(1)
dados["dia_2"] = dados["nivel"].shift(2)
dados = dados.dropna()

colunas = ["dia_1", "dia_2"]

# Avaliação em período posterior ao treinamento
corte = int(len(dados) * 0.7)
treino = dados.iloc[:corte]
teste = dados.iloc[corte:]

modelo_teste = LinearRegression()
modelo_teste.fit(treino[colunas], treino["nivel"])

previsoes_teste = modelo_teste.predict(teste[colunas])

erro = mean_absolute_error(
    teste["nivel"], previsoes_teste
) * 100

# Treinar novamente com todos os registros
modelo = LinearRegression()
modelo.fit(dados[colunas], dados["nivel"])

entrada = pd.DataFrame([{
    "dia_1": df["nivel"].iloc[-1],
    "dia_2": df["nivel"].iloc[-2]
}])

previsao = modelo.predict(entrada)[0]
proxima_data = df["data"].max() + pd.Timedelta(days=1)
mudanca = (previsao - nivel) * 100

st.divider()
st.subheader("🧠 Previsão experimental com IA")

st.metric(
    f"Estimativa para {proxima_data:%d/%m/%Y}",
    f"{previsao:.2f} m",
    f"{mudanca:.1f} cm"
)

st.write("Modelo utilizado: Regressão Linear")

st.write(
    f"Erro médio no teste histórico: {erro:.2f} cm"
    
st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:25px;">
    <p style="color:#94a3b8;">
        🌊 Idealizado e desenvolvido por
    </p>
    <h3 style="color:#38bdf8;">
        Arilan da Silva
    </h3>
    <p style="color:#94a3b8;font-size:12px;">
        © 2026 | Várzea Inteligente
    </p>
</div>
""", unsafe_allow_html=True)

)

st.warning(
    "Previsão experimental baseada em poucos dados. "
    "O erro informado foi medido em um teste limitado "
    "e não garante precisão futura. "
    "Não utilizar como alerta oficial."
)

st.caption("Fonte das medições: Porto de Manaus")
