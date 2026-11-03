import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Várzea Inteligente", page_icon="🌊", layout="centered")

st.markdown("## 🌊 Várzea Inteligente")
st.caption("Rio Negro — Manaus | Monitoramento e previsão experimental")

try:
    df = pd.read_csv("rio_negro.csv")
    if not {"data", "nivel"}.issubset(df.columns):
        st.error("O CSV precisa ter as colunas 'data' e 'nivel'.")
        st.stop()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["nivel"] = pd.to_numeric(df["nivel"], errors="coerce")
    df = df.dropna(subset=["data", "nivel"]).sort_values("data")
    df = df.drop_duplicates(subset="data", keep="last").reset_index(drop=True)
except FileNotFoundError:
    st.error("Arquivo rio_negro.csv não encontrado ao lado de app.py.")
    st.stop()

if len(df) < 12:
    st.error("São necessários pelo menos 12 registros consecutivos para esta demonstração.")
    st.stop()

if not df["data"].diff().dropna().dt.days.eq(1).all():
    st.error("A série tem dias ausentes. Corrija o CSV antes de usar o modelo.")
    st.stop()

ultimo = float(df["nivel"].iloc[-1])
inicial = float(df["nivel"].iloc[0])
variacao_cm = (ultimo - inicial) * 100
ultima_data = df["data"].max()

col1, col2 = st.columns(2)
col1.metric("Última cota da base", f"{ultimo:.2f} m")
col2.metric("Variação no período", f"{variacao_cm:+.0f} cm")

if variacao_cm < 0:
    st.info("📉 Tendência observada no período: descida")
elif variacao_cm > 0:
    st.info("📈 Tendência observada no período: subida")
else:
    st.info("➖ Sem variação entre primeiro e último registro")

st.caption(f"Último registro da base: {ultima_data:%d/%m/%Y} — não é leitura em tempo real.")

st.subheader("📊 Histórico do Rio Negro")
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(df["data"], df["nivel"], color="#1878b9", linewidth=2.5)
ax.set_ylabel("Cota (m)")
ax.set_xlabel("Data")
ax.set_ylim(df["nivel"].min() - 0.2, df["nivel"].max() + 0.2)
ax.grid(alpha=0.2)
fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("🧠 Previsão experimental com IA")
serie = df.copy()
serie["dia_1"] = serie["nivel"].shift(1)
serie["dia_2"] = serie["nivel"].shift(2)
serie = serie.dropna().reset_index(drop=True)
features = ["dia_1", "dia_2"]
corte = int(len(serie) * 0.7)
treino, teste = serie.iloc[:corte], serie.iloc[corte:]

modelo_teste = LinearRegression().fit(treino[features], treino["nivel"])
prev_teste = modelo_teste.predict(teste[features])
erro_ia = mean_absolute_error(teste["nivel"], prev_teste) * 100
erro_basico = mean_absolute_error(teste["nivel"], teste["dia_1"]) * 100

modelo = LinearRegression().fit(serie[features], serie["nivel"])
entrada = pd.DataFrame([{"dia_1": ultimo, "dia_2": float(df["nivel"].iloc[-2])}])
prev = float(modelo.predict(entrada)[0])
prox_data = ultima_data + pd.Timedelta(days=1)

st.metric(f"Estimativa para {prox_data:%d/%m/%Y}", f"{prev:.2f} m", f"{(prev - ultimo) * 100:+.1f} cm", delta_color="off")
st.caption("Modelo: Regressão Linear | Previsão de um dia à frente")
st.write(f"Erro médio no teste histórico ({len(teste)} dias): **{erro_ia:.2f} cm**")
st.write(f"Erro do método básico no mesmo teste: **{erro_basico:.2f} cm**")
st.warning("Protótipo didático com poucos dados, especialmente de vazante. Não é previsão validada para cheias ou secas e não substitui avisos oficiais.")

st.divider()
st.subheader("📅 Previsões de 1, 3 e 7 dias")

st.caption(
    "Demonstração histórica com dados até "
    "30/10/2026. Não são previsões atuais."
)

try:
    previsoes_hist = pd.read_csv(
        "previsoes_experimentais.csv"
    )

    colunas = st.columns(3)

    for coluna, dias in zip(colunas, [1, 3, 7]):
        linha = previsoes_hist[
            previsoes_hist["dias"] == dias
        ].iloc[0]

        with coluna:
            st.metric(
                f"{dias} dia(s)",
                f"{linha['previsao_m']:.3f} m"
            )
            st.caption(str(linha["data"]))

    st.warning(
        "Estimativas experimentais, sem "
        "validação para alertas oficiais."
    )

except (FileNotFoundError, KeyError, IndexError):
    st.error(
        "Arquivo de previsões não encontrado "
        "ou com formato incorreto."
    )


st.subheader("📈 Histórico e previsões da IA")

prev_grafico = pd.read_csv(
    "previsoes_experimentais.csv"
)

prev_grafico["data"] = pd.to_datetime(
    prev_grafico["data"],
    dayfirst=True
)

fig, ax = plt.subplots(figsize=(10, 4))

# Medições históricas
ax.plot(
    df["data"], df["nivel"],
    color="#1878b9",
    linewidth=2,
    label="Medições históricas"
)

# Estimativas experimentais
ax.scatter(
    prev_grafico["data"],
    prev_grafico["previsao_m"],
    color="#f97316",
    s=80,
    label="Previsões da IA",
    zorder=3
)

for _, linha in prev_grafico.iterrows():
    ax.annotate(
        f"{int(linha['dias'])}d",
        (linha["data"], linha["previsao_m"]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center"
    )

ax.set_xlabel("Data")
ax.set_ylabel("Cota do rio (m)")
ax.legend()
ax.grid(alpha=0.2)

fig.autofmt_xdate()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Linha azul: medições até 28/10/2026. "
    "Pontos laranja: estimativas experimentais "
    "calculadas com dados até 30/10/2026. "
    "As séries ainda usam bases de datas diferentes."
)

st.subheader("🧭 Informações para seu contexto")
perfil = st.selectbox("Qual é o seu perfil?", ["Morador ribeirinho", "Comerciante", "Transportador fluvial"])

if perfil == "Morador ribeirinho":
    st.write("Acompanhe os comunicados oficiais e planeje com antecedência deslocamentos e acesso a serviços essenciais conforme as condições locais.")
elif perfil == "Comerciante":
    st.write("Considere acompanhar fornecedores, prazos de transporte e estoques durante períodos de variação acentuada dos rios.")
else:
    st.write("Consulte boletins oficiais e informações locais de navegação antes de planejar rotas. A cota de Manaus não indica sozinha a navegabilidade de uma rota.")

st.caption("Essas orientações são informativas, baseadas no perfil escolhido, e não são geradas pela IA nem constituem alertas de risco.")
st.markdown("[Fonte: Porto de Manaus](https://portodemanaus.com.br/nivel-do-rio-negro/) · [Monitoramento oficial: SGB](https://sace.sgb.gov.br/amazonas/)")
st.divider()
st.markdown("<div style='text-align:center;color:#64748b'>🌊 Idealizado e desenvolvido por<br><strong style='font-size:1.2rem;color:#1683c2'>Arilan da Silva</strong><br>© 2026 · Várzea Inteligente</div>", unsafe_allow_html=True)
