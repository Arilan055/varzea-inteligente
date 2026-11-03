"""Várzea Inteligente — painel educacional (Missão 35).

Este aplicativo usa uma base histórica experimental e uma tabela de previsões
já calculadas. Não faz inferências em tempo real e não emite alertas oficiais.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Várzea Inteligente", page_icon="🌊", layout="centered")

PASTA = Path(__file__).resolve().parent
HISTORICO = PASTA / "rio_negro_treino_experimental.csv"
PREVISOES = PASTA / "previsoes_experimentais.csv"


@st.cache_data(show_spinner=False)
def carregar_historico(caminho: str) -> pd.DataFrame:
    dados = pd.read_csv(caminho)
    obrigatorias = {"data", "nivel", "corrigido"}
    ausentes = obrigatorias - set(dados.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes na base histórica: {', '.join(sorted(ausentes))}")
    dados["data"] = pd.to_datetime(dados["data"], errors="coerce")
    dados["nivel"] = pd.to_numeric(dados["nivel"], errors="coerce")
    dados["corrigido"] = dados["corrigido"].astype(str).str.strip().str.lower().isin(
        ["true", "1", "sim"]
    )
    dados = (
        dados.dropna(subset=["data", "nivel"])
        .sort_values("data")
        .drop_duplicates("data", keep="last")
        .reset_index(drop=True)
    )
    if dados.empty:
        raise ValueError("A base histórica não contém registros utilizáveis.")
    return dados


@st.cache_data(show_spinner=False)
def carregar_previsoes(caminho: str) -> pd.DataFrame:
    dados = pd.read_csv(caminho)
    obrigatorias = {"data", "dias", "previsao_m"}
    ausentes = obrigatorias - set(dados.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes no arquivo de previsões: {', '.join(sorted(ausentes))}")
    dados["data"] = pd.to_datetime(dados["data"], format="%d/%m/%Y", errors="coerce")
    dados["dias"] = pd.to_numeric(dados["dias"], errors="coerce")
    dados["previsao_m"] = pd.to_numeric(dados["previsao_m"], errors="coerce")
    dados = dados.dropna(subset=["data", "dias", "previsao_m"]).sort_values("dias")
    if dados.empty:
        raise ValueError("Arquivo de previsões vazio ou com datas inválidas.")
    return dados


st.title("🌊 Várzea Inteligente")
st.caption("Rio Negro — Manaus | Histórico e previsões experimentais")

try:
    historico = carregar_historico(str(HISTORICO))
except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
    st.error("Não foi possível carregar a base histórica da pesquisa.")
    st.code(str(exc))
    st.info("Confirme se rio_negro_treino_experimental.csv está no mesmo diretório que app.py.")
    st.stop()

ultimo_dia = historico["data"].max()
ultimos_30 = historico.loc[historico["data"] >= ultimo_dia - pd.Timedelta(days=29)].copy()
primeiro_30 = float(ultimos_30["nivel"].iloc[0])
ultimo_nivel = float(ultimos_30["nivel"].iloc[-1])
variacao_30 = 100 * (ultimo_nivel - primeiro_30)

col1, col2 = st.columns(2)
col1.metric("Última cota na base", f"{ultimo_nivel:.2f} m")
col2.metric("Variação nos últimos 30 dias da base", f"{variacao_30:+.0f} cm")
if variacao_30 > 0:
    st.info("📈 Tendência no período: subida")
elif variacao_30 < 0:
    st.info("📉 Tendência no período: descida")
else:
    st.info("➖ Sem variação líquida no período")

st.caption(
    f"Último registro da série: {ultimo_dia:%d/%m/%Y}. "
    "Base experimental, não é medição em tempo real."
)

st.subheader("📊 Últimos 30 dias da base")
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ultimos_30["data"], ultimos_30["nivel"], color="#1878b9", linewidth=2.4)
estimados_30 = ultimos_30[ultimos_30["corrigido"]]
if not estimados_30.empty:
    ax.scatter(estimados_30["data"], estimados_30["nivel"], color="#f97316", s=36, zorder=3)
ax.set_ylabel("Cota (m)")
ax.set_xlabel("Data")
ax.grid(alpha=0.22)
fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()
st.subheader("🧠 Previsões históricas experimentais — 1, 3 e 7 dias")
try:
    previsoes = carregar_previsoes(str(PREVISOES))
    # Validamos a âncora da previsão: data prevista menos o horizonte deve
    # coincidir com a data de referência do arquivo histórico.
    bases = previsoes["data"] - pd.to_timedelta(previsoes["dias"], unit="D")
    bases_validas = (bases == ultimo_dia).all()
    if not bases_validas:
        st.warning(
            "As datas das previsões não correspondem à última data da série histórica. "
            "Confira os arquivos antes de interpretar este gráfico."
        )
    else:
        st.caption(f"Estimativas calculadas usando informações até {ultimo_dia:%d/%m/%Y}.")

    colunas = st.columns(3)
    for coluna, horizonte in zip(colunas, [1, 3, 7]):
        item = previsoes[previsoes["dias"] == horizonte]
        with coluna:
            if item.empty:
                st.warning(f"Sem previsão de {horizonte} dia(s)")
            else:
                linha = item.iloc[0]
                st.metric(f"{horizonte} dia(s)", f"{linha['previsao_m']:.3f} m")
                st.caption(f"{linha['data']:%d/%m/%Y}")

    st.subheader("📈 Medições históricas e pontos previstos")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ultimos_30["data"], ultimos_30["nivel"], color="#1878b9", linewidth=2.2, label="Histórico")
    ax.scatter(previsoes["data"], previsoes["previsao_m"], color="#f97316", s=60, label="Previsões experimentais", zorder=4)
    for _, linha in previsoes.iterrows():
        ax.annotate(f"{int(linha['dias'])}d", (linha["data"], linha["previsao_m"]),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xlabel("Data")
    ax.set_ylabel("Cota (m)")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Linha azul: histórico experimental. Pontos laranja: estimativas já geradas, não medições observadas.")
except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
    st.warning("Não foi possível exibir as previsões salvas.")
    st.code(str(exc))

st.warning(
    "Os dados incluem valores interpolados e derivam de fonte secundária. "
    "As previsões apresentadas são resultados históricos de pesquisa, não alertas oficiais "
    "nem projeções atualizadas automaticamente."
)

st.divider()
st.subheader("🗓️ Explore o histórico do Rio Negro")
periodo = st.selectbox("Escolha o período", [
    "Últimos 30 dias da base", "Últimos 12 meses da base", "Selecionar ano", "Todo o histórico"
])
if periodo == "Últimos 30 dias da base":
    filtrado = ultimos_30
elif periodo == "Últimos 12 meses da base":
    inicio = ultimo_dia - pd.DateOffset(years=1)
    filtrado = historico[historico["data"] >= inicio]
elif periodo == "Selecionar ano":
    anos = sorted(historico["data"].dt.year.unique().tolist(), reverse=True)
    ano = st.selectbox("Escolha o ano", anos)
    filtrado = historico[historico["data"].dt.year == ano]
else:
    filtrado = historico

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(filtrado["data"], filtrado["nivel"], color="#1683c2", linewidth=1.6, label="Série histórica tratada")
interp = filtrado[filtrado["corrigido"]]
if not interp.empty:
    ax.scatter(interp["data"], interp["nivel"], color="#f97316", s=23, label="Valores estimados", zorder=3)
ax.set_xlabel("Data")
ax.set_ylabel("Cota (m)")
ax.grid(alpha=0.22)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption(f"Período: {filtrado['data'].min():%d/%m/%Y} até {filtrado['data'].max():%d/%m/%Y}")
st.metric("Dias exibidos", len(filtrado))
st.caption(f"Valores estimados no período: {int(filtrado['corrigido'].sum())}.")

st.divider()
st.subheader("🧭 Informações para seu contexto")
perfil = st.selectbox("Qual é o seu perfil?", ["Morador ribeirinho", "Comerciante", "Transportador fluvial"])
if perfil == "Morador ribeirinho":
    st.write("Acompanhe comunicados oficiais e planeje deslocamentos e acesso a serviços essenciais conforme as condições locais.")
elif perfil == "Comerciante":
    st.write("Considere monitorar fornecedores, prazos de transporte e estoques durante períodos de variação acentuada dos rios.")
else:
    st.write("Consulte boletins oficiais e informações locais de navegação antes de planejar rotas. A cota de Manaus, isoladamente, não determina a navegabilidade.")
st.caption("Orientações gerais por perfil, não produzidas pela IA e não equivalentes a alertas de risco.")
st.markdown("[Fonte: Porto de Manaus](https://portodemanaus.com.br/nivel-do-rio-negro/) · [Monitoramento oficial: SGB](https://sace.sgb.gov.br/amazonas/)")
st.divider()
st.markdown("<div style='text-align:center;color:#64748b'>🌊 Idealizado e desenvolvido por<br><strong style='font-size:1.2rem;color:#1683c2'>Arilan da Silva</strong><br>© 2026 · Várzea Inteligente</div>", unsafe_allow_html=True)
