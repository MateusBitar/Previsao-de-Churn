# Plano do projeto — Previsão de Churn (Telco)

Documento de planejamento alinhado ao desafio (leitura do problema, etapas ordenadas, critérios de sucesso). Reflete **o que foi executado** no repositório.

---

## Leitura do problema e objetivo

- **Contexto:** base tabular de clientes de telecom (Kaggle *Telco Customer Churn*) com perfil, serviços, faturamento e indicador de cancelamento.
- **Problema de negócio:** identificar clientes com maior chance de churn para ações de **retenção proativas**; em churn, o custo de **não detectar** um cancelador (falso negativo) costuma ser alto.
- **Objetivo técnico:** explorar os dados, preparar features, treinar e comparar modelos de classificação binária, expor **probabilidade de churn** como score de risco e disponibilizar inferência em interface simples (Streamlit).

---

## Etapas em ordem (como foram realizadas)

| Ordem | Etapa | O que foi feito |
|------:|--------|-----------------|
| 1 | **Compreensão do dataset e setup** | Definir fonte (`blastchar/telco-customer-churn`), ambiente Python, dependências principais. |
| 2 | **EDA inicial** | `df.info()`, distribuição básica, correlação das variáveis com `Churn`, gráfico de barras (Pearson), top fatores positivos/negativos. |
| 3 | **Limpeza e encoding** | Mesma lógica em **`telco_preprocess.py`** (`preprocess_telco_raw`): `TotalCharges` numérico + drop de NaNs; remover `customerID`; mapear alvo e binárias; one-hot (`get_dummies`, `drop_first`). |
| 4 | **Split treino/teste** | 80/20, `random_state=42`, mantendo conjunto de teste com proporção real de churn. |
| 5 | **Baseline modelagem** | `RandomForestClassifier` com `class_weight='balanced'`; `classification_report` e matriz de confusão. |
| 6 | **Tratamento de desbalanceamento** | **SMOTE** apenas em `X_train, y_train`; novo Random Forest sem `class_weight`; reavaliação no mesmo teste. |
| 7 | **Modelo avançado** | **XGBoost** treinado nos dados de treino pós-SMOTE; métricas e matriz de confusão; comparação com etapas anteriores. |
| 8 | **Interpretabilidade** | `feature_importances_` (top variáveis), alinhamento com insights de negócio no app. |
| 9 | **Persistência e app** | `joblib` do modelo XGBoost + `colunas_treino`; Streamlit com quatro abas — Simulador, **Predição em lote (CSV)**, Insights (importâncias), Bastidores (recall/SMOTE/XGBoost e matrizes). Carregamento dos `.joblib` ao lado de `app.py`. |
| 10 | **Documentação** | README com contexto, stack, como rodar localmente e link de deploy (quando aplicável). |

---

## Critérios de sucesso (definidos para este projeto)

1. **Métrica principal:** **recall da classe churn** no conjunto de teste — priorizar detectar cancelamentos reais, aceitando trade-off com precision.
2. **Generalização:** avaliação sempre no **holdout 20%** que não passou por SMOTE; SMOTE **somente** no treino.
3. **Comparação de abordagens:** pelo menos duas linhas claras — Random Forest (baseline e/ou com balanceamento de classes) vs **XGBoost + SMOTE** como candidato final.
4. **Score de risco:** probabilidade de churn (`predict_proba` classe positiva), interpretável como “% de risco” na interface.
5. **Reprodutibilidade:** artefatos salvos (`modelo_churn_xgboost.joblib`, `colunas_treino.joblib`) e mesma ordem de colunas na inferência (`align_to_training_columns` / `features_for_model` em `telco_preprocess.py`).
6. **Produto mínimo:** app Streamlit funcional para demonstração ao vivo.

---

## Observação (aderência plano × entrega)

O núcleo analítico e de modelagem seguiu as etapas acima. Itens adicionais alinhados ao edital (notebook de EDA, relatório, upload em lote no Streamlit, etc.) estão descritos no **`plano_incremento.md`**.
