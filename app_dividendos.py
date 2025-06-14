import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Configurações da Carteira (Você pode editar isso!) ---
CARTEIRA_ACOES = {
    "BBAS3.SA": {"nome": "Banco do Brasil", "setor": "Bancos", "peso": 0.10},
    "ITUB4.SA": {"nome": "Itaú Unibanco", "setor": "Bancos", "peso": 0.10},
    "VALE3.SA": {"nome": "Vale", "setor": "Mineração & Siderurgia", "peso": 0.10},
    "PETR4.SA": {"nome": "Petrobras", "setor": "Petróleo & Gás", "peso": 0.10},
    "WEGE3.SA": {"nome": "WEG", "setor": "Bens de Capital", "peso": 0.10},
    "MGLU3.SA": {"nome": "Magazine Luiza", "setor": "Varejo", "peso": 0.10}, # Exemplo adicional
    "SUZB3.SA": {"nome": "Suzano", "setor": "Papel e Celulose", "peso": 0.10}, # Exemplo adicional
    "RENT3.SA": {"nome": "Localiza", "setor": "Serviços", "peso": 0.10}, # Exemplo adicional
    "PRIO3.SA": {"nome": "PRIO", "setor": "Petróleo & Gás", "peso": 0.10}, # Exemplo adicional
    "B3SA3.SA": {"nome": "B3", "setor": "Serviços Financeiros", "peso": 0.10}, # Exemplo adicional
}

# Ticker do benchmark (Ibovespa)
TICKER_IBOV = "^BVSP"

# Período de análise (últimos 2 anos para um bom histórico)
DIAS_HISTORICO = 730 # Aproximadamente 2 anos

# --- Título e Introdução ---
st.title("Carteira Dividendos - Análise Detalhada")
st.write("Esta aplicação simula a análise de uma carteira de ações focada em dividendos, apresentando informações detalhadas e desempenho histórico.")

st.markdown("""
Para o investidor em busca de ações com boa perspectiva de distribuição contínua de rendimento através de dividendos. O investimento nesses nomes é uma alternativa para quem busca menor volatilidade no valor das ações e oportunidade de criar um fluxo de caixa recorrente por meio da distribuição dos lucros pelas companhias.
""")

# --- Função para buscar dados de ações ---
@st.cache_data
def buscar_dados_acao(ticker):
    acao = yf.Ticker(ticker)
    info = acao.info
    hoje = datetime.now()
    inicio = hoje - timedelta(days=DIAS_HISTORICO)
    hist = acao.history(start=inicio, end=hoje)
    dividends = acao.dividends.loc[str(inicio.year):str(hoje.year)]
    return info, hist, dividends

# --- Carregar dados da carteira ---
st.subheader("Composição da Carteira")
dados_carteira = []
precos_fechamento = pd.DataFrame()
dividend_yields = {}

with st.spinner("Carregando dados das ações da carteira..."):
    for ticker, atributos in CARTEIRA_ACOES.items():
        try:
            info, hist, dividends = buscar_dados_acao(ticker)
            
            # Adicionar preço de fechamento ao DataFrame combinado
            if not hist.empty:
                precos_fechamento[ticker] = hist['Close']

            # Calcular Dividend Yield (últimos 12 meses)
            hoje = datetime.now()
            um_ano_atras = hoje - timedelta(days=365)
            dividends_12m = dividends.loc[str(um_ano_atras.year):str(hoje.year)].sum()
            
            current_price = info.get('currentPrice')
            
            if current_price and current_price > 0:
                dy = (dividends_12m / current_price) * 100
                dividend_yields[ticker] = dy
            else:
                dy = 0 # ou 'N/A'
                dividend_yields[ticker] = dy

            dados_carteira.append({
                "Companhia": atributos["nome"],
                "Ticker": ticker,
                "Peso": f"{atributos['peso']*100:.0f}%",
                "Setor": atributos["setor"],
                "Dividend Yield": f"{dy:.2f}%" if current_price else "N/A"
            })
        except Exception as e:
            st.warning(f"Não foi possível carregar dados para {ticker}: {e}")
            dados_carteira.append({
                "Companhia": atributos["nome"],
                "Ticker": ticker,
                "Peso": f"{atributos['peso']*100:.0f}%",
                "Setor": atributos["setor"],
                "Dividend Yield": "Erro"
            })

df_carteira = pd.DataFrame(dados_carteira)
df_carteira_sorted = df_carteira.sort_values(by="Companhia").reset_index(drop=True)
st.dataframe(df_carteira_sorted, use_container_width=True)

# --- Desempenho da Carteira vs. Benchmark ---
st.subheader("Desempenho da Carteira vs. Benchmark (Ibovespa)")

if not precos_fechamento.empty:
    with st.spinner("Calculando desempenho da carteira..."):
        # Buscar dados do Ibovespa
        info_ibov, hist_ibov, _ = buscar_dados_acao(TICKER_IBOV)
        
        # Combinar preços (preencher NaNs com ffill ou bfill para evitar problemas)
        dados_combinados = pd.concat([precos_fechamento, hist_ibov['Close'].rename(TICKER_IBOV)], axis=1)
        dados_combinados = dados_combinados.dropna() # Remover datas onde não há dados para todos

        # Normalizar os preços para a base 100
        dados_normalizados = (dados_combinados / dados_combinados.iloc[0]) * 100

        # Calcular o retorno ponderado da carteira
        retorno_carteira = pd.Series(0.0, index=dados_normalizados.index)
        for ticker, atributos in CARTEIRA_ACOES.items():
            if ticker in dados_normalizados.columns:
                retorno_carteira += dados_normalizados[ticker] * atributos['peso']

        # Gráfico de Desempenho
        fig_desempenho, ax_desempenho = plt.subplots(figsize=(12, 6))
        ax_desempenho.plot(retorno_carteira, label="Carteira Dividendos", color='orange')
        ax_desempenho.plot(dados_normalizados[TICKER_IBOV], label="Ibovespa", color='blue')
        
        ax_desempenho.set_title("Desempenho Histórico da Carteira vs. Ibovespa (Base 100)")
        ax_desempenho.set_xlabel("Data")
        ax_desempenho.set_ylabel("Retorno (Base 100)")
        ax_desempenho.legend()
        ax_desempenho.grid(True)
        st.pyplot(fig_desempenho)

        # Resumo de retorno (simplificado)
        if not retorno_carteira.empty:
            retorno_carteira_total = (retorno_carteira.iloc[-1] / retorno_carteira.iloc[0] - 1) * 100 if retorno_carteira.iloc[0] != 0 else 0
            retorno_ibov_total = (dados_normalizados[TICKER_IBOV].iloc[-1] / dados_normalizados[TICKER_IBOV].iloc[0] - 1) * 100 if dados_normalizados[TICKER_IBOV].iloc[0] != 0 else 0
            st.write(f"**Retorno Total da Carteira no período:** {retorno_carteira_total:.2f}%")
            st.write(f"**Retorno Total do Ibovespa no período:** {retorno_ibov_total:.2f}%")
else:
    st.warning("Não foi possível carregar dados suficientes para calcular o desempenho da carteira.")

# --- Fluxo de Pagamento de Dividendos (Simplificado para o portfólio) ---
st.subheader("Fluxo de Pagamento de Dividendos da Carteira")

# Este é um cálculo simplificado. Um fluxo real exigiria mais lógica de pesos e datas de pagamento
# Vamos somar os dividendos históricos de cada ação
dividendos_combinados = pd.Series(dtype=float)
with st.spinner("Calculando fluxo de dividendos..."):
    for ticker in CARTEIRA_ACOES.keys():
        try:
            _, _, dividends = buscar_dados_acao(ticker)
            if not dividends.empty:
                # Multiplica os dividendos pelo peso da ação na carteira para uma simulação
                # Isso é uma simplificação, pois o investimento inicial afetaria o valor real
                weighted_dividends = dividends * CARTEIRA_ACOES[ticker]['peso'] 
                dividendos_combinados = dividendos_combinados.add(weighted_dividends, fill_value=0)
        except Exception as e:
            st.warning(f"Erro ao obter dividendos para {ticker}: {e}")

if not dividendos_combinados.empty:
    dividendos_combinados_df = dividendos_combinados.to_frame(name='Dividendos Recebidos').reset_index()
    dividendos_combinados_df.columns = ['Data', 'Dividendos Recebidos']
    dividendos_combinados_df['Data'] = dividendos_combinados_df['Data'].dt.date # Apenas a data
    dividendos_combinados_df = dividendos_combinados_df.sort_values(by='Data', ascending=False)
    st.dataframe(dividendos_combinados_df, use_container_width=True)
else:
    st.write("Não há dados de dividendos para esta carteira no período selecionado.")

# --- Disclaimer (Importante!) ---
st.markdown("---")
st.subheader("Informações Importantes / Disclaimer")
st.markdown("""
* **Este aplicativo é para fins demonstrativos e educacionais apenas.** Não constitui recomendação de investimento.
* Os dados são obtidos do Yahoo Finance (yfinance) e podem conter imprecisões ou atrasos.
* **Rentabilidade passada não é garantia de rentabilidade futura.**
* Sempre consulte um profissional financeiro qualificado antes de tomar decisões de investimento.
""")