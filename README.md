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

1. **Engenharia e Limpeza de Dados:** Tratamento de inconsistências, tipagem forçada e conversão de variáveis categóricas utilizando *One-Hot Encoding*.
2. **Balanceamento Sintético (SMOTE):** O dataset apresentava um forte desbalanceamento (muito mais clientes fiéis do que evasões). Para evitar que o algoritmo ignorasse a classe minoritária, implementei o SMOTE nos dados de treino, forçando a IA a mapear os padrões de cancelamento estruturalmente.
3. **Modelagem Avançada:** O modelo baseline (Random Forest) obteve apenas 46% de detecção. Com a substituição do motor por **XGBoost (Extreme Gradient Boosting)** aliado aos dados balanceados, **o Recall saltou para 64%**.
4. **Deploy Interativo:** O cérebro do modelo foi exportado via `joblib` e integrado a uma aplicação web interativa construída com **Streamlit** e **Plotly**, permitindo simulações em tempo real.

## 🚀 Impacto Financeiro (O ROI do Modelo)
Focando na métrica de **Recall**, o modelo otimizado (XGBoost + SMOTE) conseguiu detectar **39% a mais de clientes em risco real de cancelamento** comparado ao modelo base. Em um cenário real de negócios, isso significa centenas de assinaturas a mais que a equipe de atendimento pode tentar salvar todos os meses através de ofertas ou upgrades estratégicos.

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Processamento de Dados:** Pandas, Numpy
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE)
* **Visualização:** Matplotlib, Seaborn, Plotly
* **Deploy & Web App:** Streamlit, Joblib

## 💻 Como rodar este projeto localmente

```bash
# Clone este repositório
git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)

# Acesse a pasta do projeto
cd seu-repositorio

# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação Streamlit
streamlit run app.py