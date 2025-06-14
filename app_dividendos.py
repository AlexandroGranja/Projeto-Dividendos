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

# --- Inicialização do Session State ---
if 'ia_report_text' not in st.session_state:
    st.session_state.ia_report_text = None

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
@st.cache_data(ttl=timedelta(hours=6))
def buscar_dados_acao(ticker):
    acao = yf.Ticker(ticker)
    info = acao.info
    hoje = datetime.now()
    inicio = hoje - timedelta(days=DIAS_HISTORICO)
    hist = acao.history(start=inicio, end=hoje)
    dividends = acao.dividends
    return info, hist, dividends

# --- Função para calcular o crescimento anual de dividendos (CAGR) ---
def calcular_crescimento_dividendos(dividends_series, years):
    if dividends_series.empty or len(dividends_series) < 2:
        return "N/A" # Não há dados suficientes para calcular crescimento

    # Certifique-se que o índice está como datetime e é tz-naive
    if dividends_series.index.tz is not None:
        dividends_series = dividends_series.tz_localize(None)

    hoje = datetime.now()
    inicio_periodo = hoje - timedelta(days=years * 365) # Calcula a data de início do período

    # Filtra os dividendos dentro do período desejado
    dividends_no_periodo = dividends_series[dividends_series.index >= inicio_periodo]

    if dividends_no_periodo.empty:
        return "N/A" # Nenhum dividendo no período

    # Encontra o primeiro e o último ano com dividendos no período
    # Garante que temos pelo menos um ano completo de diferença para calcular CAGR
    primeiro_ano_com_dividendo = dividends_no_periodo.index.min().year
    ultimo_ano_com_dividendo = dividends_no_periodo.index.max().year

    # Soma os dividendos do primeiro e último ano com dados
    total_dividendo_inicio = dividends_no_periodo[dividends_no_periodo.index.year == primeiro_ano_com_dividendo].sum()
    total_dividendo_fim = dividends_no_periodo[dividends_no_periodo.index.year == ultimo_ano_com_dividendo].sum()

    # Evita divisão por zero ou log de zero/negativo, ou período muito curto
    if total_dividendo_inicio <= 0 or total_dividendo_fim <= 0 or (ultimo_ano_com_dividendo - primeiro_ano_com_dividendo) < 1:
        return "N/A"

    num_periodos = ultimo_ano_com_dividendo - primeiro_ano_com_dividendo

    if num_periodos > 0:
        # Fórmula do CAGR: ((Valor Final / Valor Inicial)^(1 / Num Períodos)) - 1
        cagr = ((total_dividendo_fim / total_dividendo_inicio) ** (1 / num_periodos)) - 1
        return f"{cagr * 100:.2f}%"
    else:
        return "N/A" # Apenas um ano de dados no período, não dá pra calcular crescimento

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
            if dividends.index.tz is not None:
                dividends.index = dividends.index.tz_localize(None)
            # Garantir que a seleção de datas do dividends não falhe se não houver dados no ano
            dividends_12m = dividends[(dividends.index >= um_ano_atras) & (dividends.index <= hoje)].sum()
            
            current_price = info.get('currentPrice')
            
            if current_price and current_price > 0:
                dy = (dividends_12m / current_price) * 100
                dividend_yields_dict[ticker] = dy # Armazena para uso na IA
            else:
                dy = 0 
                dividend_yields_dict[ticker] = dy

            # --- NOVAS MÉTRICAS ADICIONADAS AQUI ---
            # Usamos .get() para evitar erros se a métrica não existir para a ação
            # Formatamos como string para exibir bonito, ou "N/A" se não encontrar
            pl = info.get('forwardPE', 'N/A') # Preço/Lucro Futuro (mais comum para análise)
            pvp = info.get('priceToBook', 'N/A') # Preço/Valor Patrimonial
            roe = info.get('returnOnEquity', 'N/A') # Retorno sobre Patrimônio Líquido
            market_cap = info.get('marketCap', 'N/A') # Capitalização de Mercado
            
            # Formatação para exibição:
            pl_display = f"{pl:.2f}" if isinstance(pl, (int, float)) else "N/A"
            pvp_display = f"{pvp:.2f}" if isinstance(pvp, (int, float)) else "N/A"
            roe_display = f"{roe*100:.2f}%" if isinstance(roe, (int, float)) else "N/A" # ROE geralmente em %
            
            # Formatação de Market Cap para bilhões (B) ou milhões (M)
            market_cap_display = "N/A"
            if isinstance(market_cap, (int, float)):
                if market_cap >= 1_000_000_000_000: # Trilhões
                    market_cap_display = f"R$ {market_cap / 1_000_000_000_000:.2f}T"
                elif market_cap >= 1_000_000_000: # Bilhões
                    market_cap_display = f"R$ {market_cap / 1_000_000_000:.2f}B"
                elif market_cap >= 1_000_000: # Milhões
                    market_cap_display = f"R$ {market_cap / 1_000_000:.2f}M"
                else:
                    market_cap_display = f"R$ {market_cap:,.2f}" # Para valores menores

             # --- NOVAS MÉTRICAS DE CRESCIMENTO DE DIVIDENDOS AQUI ---
            # Calcula o crescimento para 3 e 5 anos
            crescimento_dy_3a = calcular_crescimento_dividendos(dividends, 3)
            crescimento_dy_5a = calcular_crescimento_dividendos(dividends, 5)

            dados_carteira.append({
                "Companhia": nome_acao,
                "Ticker": ticker,
                "Peso": f"{atributos['peso']*100:.0f}%",
                "Setor": setor_acao,
                "Dividend Yield": f"{dy:.2f}%" if current_price else "N/A",
                "Preço Atual": f"R$ {current_price:.2f}" if current_price else "N/A",
                "P/L": pl_display,
                "P/VP": pvp_display,
                "ROE": roe_display,
                "Market Cap": market_cap_display,
                "Cresc. DY (3a)": crescimento_dy_3a, # <-- NOVA COLUNA AQUI
                "Cresc. DY (5a)": crescimento_dy_5a  # <-- NOVA COLUNA AQUI
            })
        except Exception as e:
            st.warning(f"Não foi possível carregar dados para {ticker}: {e}")
            # Certifique-se de que o bloco 'except' também inclua as novas colunas
            dados_carteira.append({
                "Companhia": ticker,
                "Ticker": ticker,
                "Peso": f"{atributos['peso']*100:.0f}%",
                "Setor": "N/A (Erro ao carregar)",
                "Dividend Yield": "N/A (Erro)",
                "Preço Atual": "N/A (Erro)",
                "P/L": "N/A (Erro)",
                "P/VP": "N/A (Erro)",
                "ROE": "N/A (Erro)",
                "Market Cap": "N/A (Erro)",
                "Cresc. DY (3a)": "N/A (Erro)", # <-- Adicione aqui também!
                "Cresc. DY (5a)": "N/A (Erro)"  # <-- Adicione aqui também!
            })

df_carteira = pd.DataFrame(dados_carteira)
df_carteira_sorted = df_carteira.sort_values(by="Companhia").reset_index(drop=True)
st.dataframe(df_carteira_sorted, use_container_width=True)

# --- Gráfico de Diversificação Setorial ---
st.subheader("Diversificação Setorial da Carteira")

if not df_carteira_sorted.empty:
    # Para o gráfico, precisamos do peso em formato numérico
    # Use a coluna 'Peso_Decimal' que já criamos para o prompt da IA, se ela estiver disponível
    # Caso contrário, crie-a aqui
    if 'Peso_Decimal' not in df_carteira_sorted.columns:
        df_carteira_sorted['Peso_Decimal'] = df_carteira_sorted['Peso'].str.replace('%', '').astype(float) / 100

    # Agrupa por setor e soma os pesos
    df_setores = df_carteira_sorted.groupby('Setor')['Peso_Decimal'].sum().reset_index()
    df_setores.columns = ['Setor', 'Peso Total']

    # Filtra setores com peso zero (se houver, para não aparecer no gráfico)
    df_setores = df_setores[df_setores['Peso Total'] > 0]

    if not df_setores.empty:
        fig_setor, ax_setor = plt.subplots(figsize=(8, 8))
        
        # Cria o gráfico de pizza
        wedges, texts, autotexts = ax_setor.pie(
            df_setores['Peso Total'],
            labels=[f"{s} ({p*100:.1f}%)" for s, p in zip(df_setores['Setor'], df_setores['Peso Total'])], # Exibe setor e %
            autopct='', # Remove o autopct padrão, pois já colocamos no label
            startangle=90,
            pctdistance=0.85 # Distância dos textos de porcentagem do centro
        )
        
        # Ajusta a posição dos textos para evitar sobreposição
        for autotext in autotexts:
            autotext.set_color('white') # Cor do texto da porcentagem
            autotext.set_fontsize(10)
        
        ax_setor.set_title("Distribuição da Carteira por Setor")
        ax_setor.axis('equal')  # Garante que o gráfico de pizza seja circular.
        
        st.pyplot(fig_setor)
    else:
        st.warning("Não foi possível gerar o gráfico de setores. Verifique se há dados de setor válidos.")
else:
    st.warning("Nenhum dado da carteira disponível para gerar o gráfico de setores.")

    # --- Gráfico de Comparação de Dividend Yield por Ação ---
st.subheader("Dividend Yield por Ação")

if dividend_yields_dict: # Verifica se o dicionário de DYs não está vazio
    # Converte o dicionário em DataFrame e ordena para melhor visualização
    df_dy_individual = pd.DataFrame(list(dividend_yields_dict.items()), columns=['Ticker', 'Dividend Yield'])
    df_dy_individual = df_dy_individual.sort_values(by='Dividend Yield', ascending=False)

    fig_dy, ax_dy = plt.subplots(figsize=(12, 6))
    
    # Cria o gráfico de barras
    bars = ax_dy.bar(df_dy_individual['Ticker'], df_dy_individual['Dividend Yield'], color='teal')
    
    ax_dy.set_title("Dividend Yield Individual das Ações na Carteira")
    ax_dy.set_xlabel("Ticker")
    ax_dy.set_ylabel("Dividend Yield (%)")
    ax_dy.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adiciona os valores nas barras para facilitar a leitura
    for bar in bars:
        yval = bar.get_height()
        ax_dy.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45, ha='right') # Rotaciona os rótulos do eixo X para melhor legibilidade
    plt.tight_layout() # Ajusta o layout para evitar sobreposição
    st.pyplot(fig_dy)
else:
    st.warning("Não há dados de Dividend Yield individual para gerar o gráfico.")

# --- Sugestão de Rebalanceamento da Carteira (Pesos Iguais) ---
st.subheader("Sugestão de Rebalanceamento (Pesos Iguais)")

if not df_carteira.empty:
    num_acoes = len(df_carteira)
    if num_acoes > 0:
        peso_ideal_igual = 1.0 / num_acoes # Calcula o peso igual para cada ação
        
        # Cria um DataFrame para a sugestão de rebalanceamento
        df_rebalanceamento = df_carteira[['Companhia', 'Ticker', 'Peso']].copy()
        df_rebalanceamento['Peso Atual (%)'] = df_rebalanceamento['Peso'].str.replace('%', '').astype(float)
        df_rebalanceamento['Peso Sugerido (%)'] = peso_ideal_igual * 100
        
        # Calcula a diferença para rebalancear
        df_rebalanceamento['Diferença (%)'] = df_rebalanceamento['Peso Sugerido (%)'] - df_rebalanceamento['Peso Atual (%)']
        
        st.write("Aqui está uma sugestão de como rebalancear sua carteira para que cada ação tenha um peso igual:")
        st.dataframe(df_rebalanceamento[['Companhia', 'Ticker', 'Peso Atual (%)', 'Peso Sugerido (%)', 'Diferença (%)']], use_container_width=True)

        st.info(f"**Rebalanceamento para pesos iguais**: Cada uma das {num_acoes} ações teria um peso de {peso_ideal_igual*100:.2f}% na carteira.")
        st.markdown(
            """
            * **Diferença (%) positiva:** Você precisaria **aumentar** a posição nesta ação.
            * **Diferença (%) negativa:** Você precisaria **diminuir** a posição nesta ação.
            """
        )
    else:
        st.warning("Não há ações na carteira para sugerir um rebalanceamento.")
else:
    st.warning("Nenhum dado da carteira disponível para sugerir rebalanceamento.")

    # Continuação do bloco 'Sugestão de Rebalanceamento'
if not df_carteira.empty:
    num_acoes = len(df_carteira)
    if num_acoes > 0:
        peso_ideal_igual = 1.0 / num_acoes
        
        df_rebalanceamento = df_carteira[['Companhia', 'Ticker', 'Peso']].copy()
        df_rebalanceamento['Peso Atual (%)'] = df_rebalanceamento['Peso'].str.replace('%', '').astype(float)
        df_rebalanceamento['Peso Sugerido (%)'] = peso_ideal_igual * 100
        df_rebalanceamento['Diferença (%)'] = df_rebalanceamento['Peso Sugerido (%)'] - df_rebalanceamento['Peso Atual (%)']
        
        st.write("Aqui está uma sugestão de como rebalancear sua carteira para que cada ação tenha um peso igual:")
        st.dataframe(df_rebalanceamento[['Companhia', 'Ticker', 'Peso Atual (%)', 'Peso Sugerido (%)', 'Diferença (%)']], use_container_width=True)

        st.info(f"**Rebalanceamento para pesos iguais**: Cada uma das {num_acoes} ações teria um peso de {peso_ideal_igual*100:.2f}% na carteira.")
        st.markdown(
            """
            * **Diferença (%) positiva:** Você precisaria **aumentar** a posição nesta ação.
            * **Diferença (%) negativa:** Você precisaria **diminuir** a posição nesta ação.
            """
        )

        # --- Ilustração do Máximo Dividend Yield Teórico ---
        st.subheader("Análise de Maximização de Dividend Yield (Teórico)")
        if dividend_yields_dict:
            # Encontra a ação com o maior Dividend Yield
            ticker_maior_dy = max(dividend_yields_dict, key=dividend_yields_dict.get)
            dy_maior_dy = dividend_yields_dict[ticker_maior_dy]

            st.markdown(f"""
            Para fins de ilustração teórica de maximização de Dividend Yield, se toda a sua carteira fosse alocada em apenas uma ação (concentração de 100% em um único ativo), a ação com o maior Dividend Yield atual é **{ticker_maior_dy}** com **{dy_maior_dy:.2f}%**.
            """)
            st.warning("""
            **Atenção:** Concentrar 100% da carteira em uma única ação aumenta drasticamente o risco do seu portfólio e não é uma estratégia de investimento recomendada por profissionais financeiros para a maioria dos investidores. Esta é uma informação meramente ilustrativa.
            """)
        else:
            st.warning("Não há dados de Dividend Yield para calcular a maximização teórica.")

    else:
        st.warning("Não há ações na carteira para sugerir um rebalanceamento.")
else:
    st.warning("Nenhum dado da carteira disponível para sugerir rebalanceamento.")

    # --- Exportar Dados ---
st.subheader("Exportar Dados da Carteira")

if not df_carteira.empty:
    # Converte o DataFrame para CSV
    csv_file = df_carteira_sorted.to_csv(index=False, sep=';', decimal=',') # Usar ; como separador e , como decimal para compatibilidade com Excel BR

    st.download_button(
        label="Baixar Tabela da Carteira (.csv)",
        data=csv_file,
        file_name=f"carteira_dividendos_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        help="Baixa a tabela de composição da carteira com todas as métricas em formato CSV."
    )

    # Botão para baixar o relatório da IA (aparece apenas se houver relatório)
    if st.session_state.ia_report_text:
        st.download_button(
            label="Baixar Relatório da IA (.txt)",
            data=st.session_state.ia_report_text,
            file_name=f"relatorio_ia_dividendos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            help="Baixa a análise e sugestões geradas pela Inteligência Artificial em formato de texto."
        )
    else:
        st.info("Gere a análise da IA (botão acima) para poder baixá-la.")

else:
    st.info("Nenhum dado da carteira disponível para exportação.")

# --- Função para Gerar Prompt da IA ---
def gerar_prompt_ia(df_carteira, precos_fechamento, dividend_yields_dict):
    # Formatar os dados da carteira para o prompt da IA
    carteira_markdown = df_carteira.to_markdown(index=False)

    # Adicionar o Dividend Yield individual no formato do prompt
    dy_individual_str = "\n".join([f"- {ticker}: {dy:.2f}%" for ticker, dy in dividend_yields_dict.items()])

    # ... (o cálculo do dy_ponderado_final deve vir aqui, como já está no seu código) ...
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

    **Data da Análise:** {datetime.now().strftime("%d de %B de %Y")} # <--- ADICIONE ESTA LINHA AQUI!

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
                st.session_state.ia_report_text = response.text
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