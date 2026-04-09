# 📊 Retenção Inteligente: Previsão de Churn com XGBoost

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://previsao-de-churn.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-1761c2?style=flat&logo=xgboost&logoColor=white)]()

Uma aplicação completa de Machine Learning de ponta a ponta projetada para identificar evasão de clientes (Churn) em empresas de telecomunicação, permitindo ações de retenção proativas e maximização de receita.

## 🎯 O Problema de Negócio
Em mercados competitivos, reter um cliente é significativamente mais barato do que adquirir um novo. O objetivo deste projeto foi desenvolver um pipeline de dados e um modelo preditivo capaz de identificar quais clientes têm maior probabilidade de cancelar suas assinaturas, entregando essa inteligência em uma interface acessível para as equipes de Vendas e Retenção.

## 💡 Principais Insights Exploratórios
Através da Análise Exploratória de Dados (EDA), identificamos os principais ofensores da retenção:
* **Falta de Fidelidade:** Contratos mensais (`Month-to-month`) são o maior gatilho para o cancelamento.
* **Problemas no Serviço Premium:** Clientes com internet de Fibra Óptica apresentam uma taxa de evasão desproporcionalmente alta, indicando possíveis falhas de qualidade ou precificação frente à concorrência.
* **Retenção Natural:** O tempo de casa (`tenure`) atua como um escudo; quanto mais antigo o cliente, menor a probabilidade de churn.

## ⚙️ Desenvolvimento e Arquitetura da Solução
A solução foi arquitetada focando em automação de ponta a ponta, desde a limpeza dos dados brutos até o deploy do modelo em produção:

1. **Engenharia e Limpeza de Dados:** Tratamento de inconsistências, tipagem forçada e *One-Hot Encoding*; a lógica compartilhada entre treino e inferência está em **`telco_preprocess.py`** (`preprocess_telco_raw`, `features_for_model`).
2. **Balanceamento Sintético (SMOTE):** O dataset apresentava um forte desbalanceamento (muito mais clientes fiéis do que evasões). Para evitar que o algoritmo ignorasse a classe minoritária, implementei o SMOTE nos dados de treino, forçando a IA a mapear os padrões de cancelamento estruturalmente.
3. **Modelagem Avançada:** O modelo baseline (Random Forest) obteve apenas 46% de detecção. Com a substituição do motor por **XGBoost (Extreme Gradient Boosting)** aliado aos dados balanceados, **o Recall saltou para 64%**.
4. **Deploy Interativo:** O modelo foi exportado via `joblib` e integrado a um app **Streamlit** com **Plotly**: simulador de cliente, **predição em lote por CSV** (`churn_pred`, `risk_score`), gráfico de importâncias e aba técnica com matrizes de confusão de referência.

## 🚀 Impacto Financeiro (O ROI do Modelo)
Focando na métrica de **Recall**, o modelo otimizado (XGBoost + SMOTE) conseguiu detectar **39% a mais de clientes em risco real de cancelamento** comparado ao modelo base. Em um cenário real de negócios, isso significa centenas de assinaturas a mais que a equipe de atendimento pode tentar salvar todos os meses através de ofertas ou upgrades estratégicos.

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Processamento de Dados:** Pandas, Numpy
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE)
* **Visualização:** Matplotlib, Seaborn, Plotly
* **Deploy & Web App:** Streamlit, Joblib

## 📋 Mapa das entregas (desafio / processo seletivo)

| Entrega | Onde encontrar |
|---------|----------------|
| Plano (antes do código) | [`PLANO.md`](PLANO.md) |
| EDA em notebook | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (correlações adicionais em [`churn.py`](churn.py)) |
| Pipeline de features (justificativa + código) | [`RELATORIO_FINAL.md`](RELATORIO_FINAL.md) · módulo compartilhado [`telco_preprocess.py`](telco_preprocess.py) · treino [`churn.py`](churn.py) · inferência [`app.py`](app.py) |
| Modelos comparados (≥2 abordagens) | [`churn.py`](churn.py) — Random Forest, RF+SMOTE, XGBoost+SMOTE |
| Score de risco contínuo | Probabilidade `predict_proba` → `risk_score` / simulador no Streamlit |
| Plataforma de inferência (CSV + UI) | [`app.py`](app.py) — abas Simulador e **Predição em lote (CSV)** |
| Relatório final | [`RELATORIO_FINAL.md`](RELATORIO_FINAL.md) |
| Plano de incremento / fechamento | [`plano_incremento.md`](plano_incremento.md) |

## 📓 Análise exploratória (EDA)

Notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

1. Baixe `WA_Fn-UseC_-Telco-Customer-Churn.csv` no [dataset Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
2. Coloque o arquivo na **raiz do projeto** ou na pasta **`notebooks/`**.
3. Abra o `.ipynb` no Jupyter, VS Code ou Cursor e execute as células.  
   *Se o kernel não achar o CSV* (diretório de trabalho diferente), coloque o arquivo na **raiz do repo** ou defina `TELCO_CSV_PATH` com o caminho completo do `.csv` (a primeira célula de código sobe até 20 pastas procurando o arquivo).

## 💻 Como rodar este projeto localmente

```bash
# Clone este repositório
git clone [git clone https://github.com/MateusBitar/Previsao-de-Churn.git](https://github.com/MateusBitar/Previsao-de-Churn.git)

# Acesse a pasta do projeto
cd seu-repositorio

# Instale as dependências
pip install -r requirements.txt

# Treine o modelo e gere os artefatos (modelo_churn_xgboost.joblib, colunas_treino.joblib)
python churn.py

# Execute a aplicação Streamlit (use o mesmo Python do pip install)
python -m streamlit run app.py
```

Na interface: use o **Simulador** ou a aba **Predição em lote (CSV)** com arquivo no formato [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (colunas `customerID` e `Churn` opcionais). Saída: `churn_pred` e `risk_score` (probabilidade 0–1), com download do resultado. Os artefatos `.joblib` devem estar na **mesma pasta** que `app.py` (o app resolve o caminho automaticamente).

Substitua o URL do `git clone` acima pelo repositório real quando publicar no GitHub.

Documentação: `PLANO.md`, `RELATORIO_FINAL.md`, `plano_incremento.md`. EDA: `notebooks/01_eda.ipynb`.