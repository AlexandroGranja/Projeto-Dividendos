# 📊 Carteira de Dividendos - Análise Detalhada com IA

Bem-vindo ao aplicativo "Carteira de Dividendos - Análise Detalhada com IA"! Esta ferramenta foi desenvolvida para ajudar investidores focados em dividendos a analisar suas carteiras de ações, visualizar informações importantes e receber sugestões personalizadas geradas por Inteligência Artificial.

## 🎯 Para quem é este aplicativo?

Este aplicativo é ideal para investidores que buscam:

* **Ações com bom fluxo de dividendos:** Aqueles que priorizam a distribuição contínua de rendimentos pelas empresas.
* **Menor volatilidade:** Investimentos que tendem a ter flutuações de preço mais suaves.
* **Geração de fluxo de caixa recorrente:** Construir uma fonte de renda passiva através dos lucros distribuídos pelas companhias.

## ✨ Funcionalidades Principais

Este aplicativo oferece as seguintes análises e recursos:

1.  **Upload Fácil da Carteira:** Carregue sua carteira de ações usando um arquivo CSV ou Excel.
2.  **Composição Detalhada da Carteira:** Visualização clara das suas ações, pesos, setores, Dividend Yield, preço atual e outras métricas financeiras (P/L, P/VP, ROE, Market Cap, Crescimento de DY em 3 e 5 anos).
3.  **Diversificação Setorial:** Gráfico de pizza intuitivo mostrando a distribuição dos seus investimentos por setor.
4.  **Dividend Yield por Ação:** Gráfico de barras comparando o Dividend Yield individual de cada ativo da sua carteira.
5.  **Sugestão de Rebalanceamento:** Ajuda a visualizar como sua carteira ficaria com pesos iguais para cada ação, facilitando decisões de rebalanceamento.
6.  **Análise Teórica de Maximização de Dividend Yield:** Identifica a ação com o maior DY na sua carteira para fins ilustrativos (com um importante aviso de risco).
7.  **Análise e Sugestões por Inteligência Artificial (IA):** Uma análise completa e contextualizada da sua carteira, gerada por IA, com pontos fortes, fracos, preocupações específicas, tendências, riscos e sugestões para fortalecer seu fluxo de dividendos.
8.  **Desempenho da Carteira vs. Benchmark:** Compare o desempenho histórico da sua carteira com o Ibovespa (benchmark do mercado brasileiro).
9.  **Fluxo de Pagamento de Dividendos:** Tabela com os dividendos recebidos, ponderados pelo peso da sua carteira.
10. **Exportação de Dados:** Baixe a tabela da sua carteira e o relatório de análise da IA em formato de texto.

## 🚀 Como Usar o Aplicativo

Siga estes passos simples para começar a usar a ferramenta:

### 1. Preparar seu Arquivo de Carteira

Você precisará de um arquivo CSV ou Excel com **duas colunas obrigatórias**:

* **`Ticker`**: O código da ação (ex: `PETR4.SA`, `ITUB4.SA`, `VALE3.SA`). Para ações brasileiras negociadas na B3, utilize o sufixo `.SA` (ex: `PETR4.SA`).
* **`Peso`**: O peso percentual da ação na sua carteira (ex: `0.20` para 20%, ou `20` se você preferir números inteiros - o aplicativo fará a normalização).

**Exemplo de arquivo Excel/CSV:**

| Ticker    | Peso |
| :-------- | :--- |
| ITUB4.SA  | 0.30 |
| BBDC4.SA  | 0.25 |
| CPLE6.SA  | 0.15 |
| PETR4.SA  | 0.20 |
| VALE3.SA  | 0.10 |

### 2. Acessar o Aplicativo

O aplicativo estará disponível através de um link fornecido (geralmente do Streamlit Cloud). Basta clicar no link para abri-lo no seu navegador.

### 3. Fazer o Upload da sua Carteira

* Na seção **"Definição da Carteira por Upload"**, clique em **"Browse files"** ou arraste seu arquivo (CSV ou XLSX) para a área indicada.
* Após o upload, o aplicativo processará os dados e começará a exibir a "Composição da Carteira" e os gráficos.

### 4. Gerar a Análise da IA

* Role a página até a seção **"Análise Automática da Carteira (Gerada por IA)"**.
* Clique no botão **"Atualizar Análise e Sugestões de IA"**.
* Aguarde alguns segundos (ou um pouco mais, dependendo da demanda e da conexão com a IA) enquanto a Inteligência Artificial gera a análise detalhada da sua carteira. O texto da análise aparecerá na tela.

### 5. Baixar Relatórios

* Na seção **"Exportar Dados da Carteira"**, você encontrará dois botões:
    * **"Baixar Tabela da Carteira (.csv)"**: Faz o download de uma planilha CSV com todos os dados tabulados da sua carteira.
    * **"Baixar Relatório da IA (.txt)"**: Este botão só aparecerá *depois* que você gerar a análise da IA (Passo 4). Clique nele para baixar o texto completo da análise gerada pela IA.

## ⚠️ Informações Importantes / Disclaimer

* **Finalidade Educacional:** Este aplicativo é desenvolvido para fins **demonstrativos e educacionais apenas**. Ele não deve ser interpretado como aconselhamento financeiro ou recomendação de investimento.
* **Fontes de Dados:** Os dados de mercado (preços, dividendos, informações de empresas) são obtidos através da biblioteca `yfinance` (Yahoo Finance) e podem conter imprecisões ou atrasos.
* **Risco de Investimento:** Rentabilidade passada não é garantia de rentabilidade futura. O investimento em ações envolve riscos e a possibilidade de perda de capital.
* **Consultoria Profissional:** **Sempre consulte um profissional financeiro qualificado** antes de tomar quaisquer decisões de investimento. A análise fornecida pela IA é uma ferramenta de apoio e não substitui a orientação de um especialista.

---

**Desenvolvido por:** Alexandro Granja
**Data da Última Atualização:** 14 de Junho de 2025
