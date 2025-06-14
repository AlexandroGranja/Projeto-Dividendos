import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import google.generativeai as genai # Importar a biblioteca do Gemini

# Ticker do benchmark (Ibovespa)
TICKER_IBOV = "^BVSP"

# Período de análise (últimos 2 anos para um bom histórico)
DIAS_HISTORICO = 730 # Aproximadamente 2 anos

# --- Configurar a API do Gemini (Lê a chave das secrets do Streamlit) ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except AttributeError:
    st.error("Chave de API do Gemini não configurada. Por favor, adicione 'GEMINI_API_KEY' nas Secrets do Streamlit Cloud.")
    st.stop() # Interrompe a execução se a chave não estiver configurada

# Inicializa o modelo da IA
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Título e Introdução ---
st.title("Carteira Dividendos - Análise Detalhada com IA")
st.write("Esta aplicação simula a análise de uma carteira de ações focada em dividendos, apresentando informações detalhadas e desempenho histórico.")

st.markdown("""
Para o investidor em busca de ações com boa perspectiva de distribuição contínua de rendimento através de dividendos. O investimento nesses nomes é uma alternativa para quem busca menor volatilidade no valor das ações e oportunidade de criar um fluxo de caixa recorrente por meio da distribuição dos lucros pelas companhias.
""")

# --- Upload da Carteira Pelo Usuário ---
st.subheader("Definição da Carteira por Upload")
st.info("Por favor, faça o upload de um arquivo CSV ou Excel com as colunas 'Ticker' e 'Peso'.")

uploaded_file = st.file_uploader("Escolha um arquivo CSV ou Excel", type=["csv", "xlsx"])

CARTEIRA_ACOES = {} # Inicializa como vazio
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        else: # Assumir .xlsx
            df_upload = pd.read_excel(uploaded_file)
        
        # Validar colunas
        if 'Ticker' in df_upload.columns and 'Peso' in df_upload.columns:
            total_pesos = 0
            for index, row in df_upload.iterrows():
                ticker = str(row['Ticker']).strip().upper()
                try:
                    peso = float(row['Peso'])
                    if peso > 0:
                        CARTEIRA_ACOES[ticker] = {"peso": peso} # Nome e setor serão buscados dinamicamente
                        total_pesos += peso
                    else:
                        st.warning(f"Ignorando ticker '{ticker}': Peso deve ser maior que zero.")
                except ValueError:
                    st.warning(f"Ignorando ticker '{ticker}': Peso inválido. Certifique-se de que é um número.")
            
            # Normalizar pesos se a soma não for 1.0 (100%) e houver ações
            if total_pesos > 0 and abs(total_pesos - 1.0) > 0.01:
                st.warning(f"A soma dos pesos é {total_pesos:.2f}. Normalizando os pesos para 100%.")
                for ticker in CARTEIRA_ACOES:
                    CARTEIRA_ACOES[ticker]["peso"] /= total_pesos
            elif total_pesos == 0 and len(CARTEIRA_ACOES) > 0: # Caso todos os pesos sejam 0, mas há tickers
                 st.error("Nenhuma ação com peso válido foi encontrada no arquivo. Certifique-se de que os pesos são números maiores que zero.")
                 CARTEIRA_ACOES = {} # Limpa a carteira para evitar processamento com erro
        else:
            st.error("O arquivo deve conter as colunas 'Ticker' e 'Peso'.")
            CARTEIRA_ACOES = {} # Limpa a carteira em caso de colunas inválidas

    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}. Verifique o formato e as colunas.")
        CARTEIRA_ACOES = {} # Limpa a carteira em caso de erro

if not CARTEIRA_ACOES: # Se a carteira ainda estiver vazia após o upload
    st.info("Aguardando upload de um arquivo válido para carregar a carteira.")
    st.stop() # Interrompe a execução se não houver carteira válida


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
dividend_yields_dict = {}

with st.spinner("Carregando dados das ações da carteira..."):
    for ticker, atributos in CARTEIRA_ACOES.items():
        try:
            info, hist, dividends = buscar_dados_acao(ticker)
            
            # Obtém nome e setor dinamicamente do Yahoo Finance
            nome_acao = info.get('longName', ticker)
            setor_acao = info.get('sector', 'N/A')
            
            # Adicionar preço de fechamento ao DataFrame combinado
            if not hist.empty:
                precos_fechamento[ticker] = hist['Close']

            # Calcular Dividend Yield (últimos 12 meses)
            hoje = datetime.now()
            um_ano_atras = hoje - timedelta(days=365)
            # Garantir que a seleção de datas do dividends não falhe se não houver dados no ano
            dividends_12m = dividends[(dividends.index >= um_ano_atras) & (dividends.index <= hoje)].sum()
            
            current_price = info.get('currentPrice')
            
            if current_price and current_price > 0:
                dy = (dividends_12m / current_price) * 100
                dividend_yields_dict[ticker] = dy # Armazena para uso na IA
            else:
                dy = 0 
                dividend_yields_dict[ticker] = dy

            dados_carteira.append({
                "Companhia": nome_acao, # Corrigido: usa a variável local nome_acao
                "Ticker": ticker,
                "Peso": f"{atributos['peso']*100:.0f}%",
                "Setor": setor_acao, # Corrigido: usa a variável local setor_acao
                "Dividend Yield": f"{dy:.2f}%" if current_price else "N/A",
                "Preço Atual": f"R$ {current_price:.2f}" if current_price else "N/A"
            })
        except Exception as e:
            st.warning(f"Não foi possível carregar dados para {ticker}: {e}")
            dados_carteira.append({
                "Companhia": ticker, # Usa o ticker como nome se der erro
                "Ticker": ticker,
                "Peso": f"{atributos['peso']*100:.0f}%",
                "Setor": "N/A (Erro ao carregar)", # Indica que não pôde carregar
                "Dividend Yield": "N/A (Erro)",
                "Preço Atual": "N/A (Erro)"
            })

df_carteira = pd.DataFrame(dados_carteira)
df_carteira_sorted = df_carteira.sort_values(by="Companhia").reset_index(drop=True)
st.dataframe(df_carteira_sorted, use_container_width=True)

# --- Função para Gerar Prompt da IA ---
def gerar_prompt_ia(df_carteira, precos_fechamento, dividend_yields_dict):
    # Formatar os dados da carteira para o prompt da IA
    carteira_markdown = df_carteira.to_markdown(index=False)
    
    # Adicionar o Dividend Yield individual no formato do prompt
    dy_individual_str = "\n".join([f"- {ticker}: {dy:.2f}%" for ticker, dy in dividend_yields_dict.items()])

    # Calcular o DY médio ponderado da carteira
    dy_ponderado = 0
    total_peso = 0
    # Converta a coluna 'Peso' para float antes de usar, pois ela vem como string "X%"
    df_temp = df_carteira.copy()
    df_temp['Peso_Decimal'] = df_temp['Peso'].str.replace('%', '').astype(float) / 100

    for index, row in df_temp.iterrows():
        ticker = row['Ticker']
        peso = row['Peso_Decimal']
        
        if ticker in dividend_yields_dict:
            dy = dividend_yields_dict[ticker]
            dy_ponderado += dy * peso
            total_peso += peso

    if total_peso > 0:
        dy_ponderado_final = dy_ponderado / total_peso
    else:
        dy_ponderado_final = 0

    prompt_content = f"""
    Eu sou um investidor focado em dividendos. Por favor, analise a seguinte carteira de ações e me forneça insights e possíveis sugestões.

    **Composição Atual da Carteira (e seus pesos normalizados):**
    {carteira_markdown}

    **Dividend Yields Individuais Recentes (últimos 12 meses):**
    {dy_individual_str}

    **Dividend Yield Ponderado da Carteira:** {dy_ponderado_final:.2f}%

    Com base nesses dados:
    1.  **Análise da Carteira Atual:**
        * Avalie a diversificação setorial. Há setores muito concentrados?
        * Comente sobre o Dividend Yield geral da carteira e dos papéis individuais.
        * Quais são os pontos fortes e fracos desta carteira do ponto de vista de dividendos?
        * Há alguma preocupação com alguma ação específica em termos de DY ou preço?
    2.  **Monitoramento e Sugestões (Baseado nos dados atuais):**
        * Considerando as ações desta carteira, quais delas se destacaram ou tiveram mudanças notáveis no DY recentemente (com base no DY atual fornecido)?
        * Se esta carteira busca maximizar dividendos consistentes, que tipo de ações (setores, características) poderiam ser consideradas para fortalecer ainda mais o fluxo de dividendos? (Não me dê tickers específicos, mas tipos de empresas).
        * Quais são as principais tendências ou riscos que um investidor de dividendos deveria estar ciente olhando para esta carteira no cenário atual?
    3.  **Formato da Resposta:** Por favor, forneça uma análise clara e concisa, como um relatório para um investidor.
    """
    return prompt_content

# --- Análise Automática da Carteira com IA ---
st.subheader("Análise Automática da Carteira (Gerada por IA)")
if st.button("Atualizar Análise e Sugestões de IA"): # Botão com novo texto
    if not df_carteira.empty:
        # Chamar a nova função que gera o prompt
        prompt = gerar_prompt_ia(df_carteira, precos_fechamento, dividend_yields_dict)
        
        with st.spinner("A IA está gerando a análise e sugestões..."):
            try:
                response = model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Erro ao chamar a IA: {e}")
                st.warning("Verifique sua chave de API e se há limites de uso ou se o modelo está acessível.")
    else:
        st.warning("Nenhum dado da carteira disponível para análise da IA. Faça o upload do arquivo da carteira.")


# --- Desempenho da Carteira vs. Benchmark (Ibovespa) ---
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

dividendos_combinados = pd.Series(dtype=float)
with st.spinner("Calculando fluxo de dividendos..."):
    for ticker in CARTEIRA_ACOES.keys():
        try:
            _, _, dividends = buscar_dados_acao(ticker)
            if not dividends.empty:
                weighted_dividends = dividends * CARTEIRA_ACOES[ticker]['peso'] 
                dividendos_combinados = dividendos_combinados.add(weighted_dividends, fill_value=0)
        except Exception as e:
            st.warning(f"Erro ao obter dividendos para {ticker}: {e}")

if not dividendos_combinados.empty:
    dividendos_combinados_df = dividendos_combinados.to_frame(name='Dividendos Recebidos').reset_index()
    dividendos_combinados_df.columns = ['Data', 'Dividendos Recebidos']
    dividendos_combinados_df['Data'] = dividendos_combinados_df['Data'].dt.date 
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