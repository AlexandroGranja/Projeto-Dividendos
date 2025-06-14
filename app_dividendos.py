import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Título da Aplicação
st.title("DividendoBot - Análise de Dividendos")
st.write("Analise ações e seus dividendos para tomar decisões de investimento.")

# Campo para o usuário inserir o ticker da ação
ticker_input = st.text_input("Digite o código da ação (ex: PETR4.SA, ITUB4.SA, TSLA):", "PETR4.SA")

# Botão para buscar dados
if st.button("Analisar Ação"):
    if ticker_input:
        try:
            # Baixar dados da ação
            acao = yf.Ticker(ticker_input)

            # Informações básicas da ação
            info = acao.info
            st.subheader(f"Informações para {ticker_input}:")
            st.write(f"Nome da Empresa: {info.get('longName', 'N/A')}")
            st.write(f"Setor: {info.get('sector', 'N/A')}")
            st.write(f"Mercado: {info.get('marketCap', 'N/A'):,.2f}")
            st.write(f"Preço Atual: {info.get('currentPrice', 'N/A'):.2f}")

            # Calcular Dividend Yield (DY)
            # Baixar histórico de preços para o cálculo do DY (últimos 12 meses)
            hoje = datetime.now()
            um_ano_atras = hoje - timedelta(days=365)
            hist = acao.history(start=um_ano_atras, end=hoje)

            dividendos_anuais = acao.dividends.loc[str(um_ano_atras.year):str(hoje.year)].sum()
            
            if info.get('currentPrice'):
                dividend_yield = (dividendos_anuais / info['currentPrice']) * 100 if info['currentPrice'] > 0 else 0
                st.write(f"Dividendos pagos nos últimos 12 meses: R$ {dividendos_anuais:.2f}")
                st.metric(label="Dividend Yield (DY)", value=f"{dividend_yield:.2f}%")
            else:
                st.write("Não foi possível calcular o Dividend Yield (preço atual não disponível).")


            # Exibir dividendos pagos historicamente
            st.subheader("Histórico de Dividendos Pagos:")
            dividends_df = acao.dividends
            if not dividends_df.empty:
                st.dataframe(dividends_df.reset_index().rename(columns={'Date': 'Data', 'Dividends': 'Valor do Dividendo'}), use_container_width=True)
                
                # Gráfico de dividendos ao longo do tempo
                fig_dividends, ax_dividends = plt.subplots(figsize=(10, 5))
                ax_dividends.plot(dividends_df.index, dividends_df.values, marker='o')
                ax_dividends.set_title(f"Dividendos Pagos por {ticker_input}")
                ax_dividends.set_xlabel("Data")
                ax_dividends.set_ylabel("Valor do Dividendo")
                ax_dividends.grid(True)
                st.pyplot(fig_dividends)
            else:
                st.write("Não há histórico de dividendos para esta ação.")

            # Gráfico de preço histórico (últimos 6 meses)
            st.subheader("Gráfico de Preço Histórico (Últimos 6 meses):")
            seis_meses_atras = hoje - timedelta(days=180)
            hist_6m = acao.history(period="6mo") # ou period="6mo"
            if not hist_6m.empty:
                fig_price, ax_price = plt.subplots(figsize=(10, 5))
                ax_price.plot(hist_6m.index, hist_6m['Close'])
                ax_price.set_title(f"Preço de Fechamento de {ticker_input}")
                ax_price.set_xlabel("Data")
                ax_price.set_ylabel("Preço de Fechamento")
                ax_price.grid(True)
                st.pyplot(fig_price)
            else:
                st.write("Não foi possível obter o histórico de preços.")

        except Exception as e:
            st.error(f"Ocorreu um erro ao buscar dados para {ticker_input}: {e}")
            st.warning("Verifique se o código da ação está correto (ex: PETR4.SA para Petrobras na B3).")
    else:
        st.warning("Por favor, digite um código de ação para analisar.")

# --- Seção de Ranking de Dividend Yield (Exemplo simplificado) ---
st.sidebar.subheader("Ranking de Ações por Dividend Yield (Exemplo)")
st.sidebar.write("Esta seção é um exemplo. Para um ranking real, você precisaria de uma lista de ações e lógica para buscar e comparar os DYs.")

# Exemplo de algumas ações para o ranking (você pode expandir esta lista)
acoes_exemplo = {
    "ITUB4.SA": "Itaú Unibanco",
    "BBAS3.SA": "Banco do Brasil",
    "VALE3.SA": "Vale",
    "WEGE3.SA": "WEG"
}

dy_data = []
for ticker, name in acoes_exemplo.items():
    try:
        acao_obj = yf.Ticker(ticker)
        info = acao_obj.info
        dividends_12m = acao_obj.dividends.loc[str((datetime.now() - timedelta(days=365)).year):str(datetime.now().year)].sum()
        current_price = info.get('currentPrice')
        
        if current_price and current_price > 0:
            dy = (dividends_12m / current_price) * 100
            dy_data.append({"Ação": name, "Ticker": ticker, "Dividend Yield": f"{dy:.2f}%", "Dividendos (12m)": f"R$ {dividends_12m:.2f}", "Preço Atual": f"R$ {current_price:.2f}"})
        else:
            dy_data.append({"Ação": name, "Ticker": ticker, "Dividend Yield": "N/A", "Dividendos (12m)": "N/A", "Preço Atual": "N/A"})
    except Exception as e:
        dy_data.append({"Ação": name, "Ticker": ticker, "Dividend Yield": "Erro", "Dividendos (12m)": "Erro", "Preço Atual": "Erro"})

if dy_data:
    df_dy = pd.DataFrame(dy_data)
    # Classifica por Dividend Yield, tentando converter para float para ordenar
    df_dy_sorted = df_dy.copy()
    df_dy_sorted['DY_Float'] = df_dy_sorted['Dividend Yield'].str.replace('%', '').replace('N/A', '0').replace('Erro', '0').astype(float)
    df_dy_sorted = df_dy_sorted.sort_values(by='DY_Float', ascending=False)
    df_dy_sorted = df_dy_sorted.drop(columns=['DY_Float']) # Remove a coluna auxiliar
    st.sidebar.dataframe(df_dy_sorted, use_container_width=True)