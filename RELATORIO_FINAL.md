# Relatório final — Previsão de Churn (Telco)

Documento de encerramento do projeto: o que funcionou, o que não funcionou e próximos passos. Alinhado ao desafio (EDA → features → modelos → score → inferência).

---

## O que funcionou

- **Leitura de negócio clara:** o foco em **recall** da classe churn (detectar quem de fato cancela) combina com uso de retenção proativa; falsos negativos são caros quando a equipe não entra em contato a tempo.
- **Pipeline de preparação coerente com dados tabulares:** conversão de `TotalCharges` para numérico com remoção de inválidos, exclusão de `customerID`, mapeamento de variáveis binárias e **one-hot encoding** (`get_dummies`, `drop_first=True`) para categorias nominais — adequado a Random Forest e XGBoost, que não exigem escala nas mesmas condições de modelos lineares simples.
- **Tratamento honesto do desbalanceamento:** **SMOTE** aplicado **apenas ao conjunto de treino**, preservando o teste com a proporção real de churn — evita otimismo artificial na avaliação.
- **Comparação de abordagens:** baseline **Random Forest** com `class_weight='balanced'`; em seguida **RF + SMOTE**; modelo final **XGBoost + SMOTE**. Na narrativa documentada no app/README, o **recall** de cancelamentos subiu da ordem de **~46%** (baseline) para **~64%** (XGBoost + SMOTE), ampliando a capacidade de “pegar” evasões reais no holdout.
- **Score de risco contínuo:** uso da **probabilidade estimada de churn** (`predict_proba`, classe positiva), exposta na interface como percentual — simples, interpretável e alinhada ao modelo escolhido.
- **Reprodutibilidade na inferência:** persistência com `joblib` do modelo e da **lista ordenada de colunas** do treino; na predição, `reindex` com `fill_value=0` garante o mesmo esquema de features do treinamento.
- **Interpretabilidade para negócio:** `feature_importances_` do XGBoost e texto no Streamlit conectam decisões do modelo a variáveis conhecidas (ex.: contrato mensal, fibra, `tenure`), em linha com a EDA.

---

## O que não funcionou ou ficou aquém

- **Trade-off de precision:** ao priorizar recall com SMOTE e boosting, é esperado **aumento de falsos positivos** (clientes marcados em risco que não cancelariam), o que pode gerar custo de campanha ou fadiga de contato — o projeto documenta o ganho em recall, mas **não calibrou nem otimizou threshold por custo** (ex.: matriz de custo FP vs FN).
- **Avaliação com um único split:** treino/teste 80/20 com `random_state` fixo dá uma estimativa única; **não há validação cruzada estratificada** nem intervalo de confiança para as métricas — a comparação entre modelos é honesta dentro desse split, mas a variância entre splits não foi quantificada.
- **Calibração de probabilidades:** o score é a saída bruta do XGBoost; **não foi verificada calibração** (curvas de calibração, Brier score, isotônica/Platt). O percentual é útil para **ordenação** de risco, mas “72%” não foi validado como frequência real de churn nessa faixa.
- **Overfitting / generalização:** não há relatório sistemático **treino vs teste** (gap de métricas) no repositório; árvores profundas e SMOTE podem, em bases menores, empurrar métricas de treino — ponto a explicitar em apresentação e a reforçar com validação mais rígida depois.
- **Interpretabilidade vs caixa-preta:** XGBoost é mais forte que um modelo linear, porém **menos transparente** que regressão logística com coeficientes diretos; a explicação ficou centrada em importância de features, não em efeitos marginais isolados.
- **EDA em dois lugares:** o notebook `notebooks/01_eda.ipynb` cobre distribuição do alvo e gráficos-chave; o script `churn.py` ainda concentra correlações e treino — dá para unificar mais adiante se quiser um único fluxo.

---

## Pipeline de features (resumo justificado)

1. **Limpeza:** `TotalCharges` como numérico; linhas sem valor válido removidas; `customerID` removida (sem sinal preditivo).
2. **Alvo e binárias:** `Churn` e campos Yes/No (e gênero) mapeados a 0/1.
3. **Categóricas restantes:** one-hot com `drop_first=True` para evitar redundância entre dummies.
4. **Modelagem:** split temporalmente neutro (shuffle) 80/20; SMOTE **somente** em `X_train, y_train`.
5. **Inferência:** mesmo encadeamento lógico de encoding + alinhamento de colunas ao artefato salvo.

*Não foi aplicada padronização numérica global porque o modelo final é baseado em árvores, onde monotonicidade por escala linear não é requisito como em muitos modelos paramétricos.*

---

## O que eu faria com mais tempo, mais dados ou mais contexto de negócio

- **Validação:** k-fold estratificado e/ou repeated holdout para métricas mais estáveis; busca de hiperparâmetros documentada (optuna/random search).
- **Threshold e custos:** definir cutoff por **custo esperado** de FP/FN ou orçamento de contatos da operação de retenção.
- **Calibração e monitoramento:** calibrar probabilidades; em produção, **drift** de features e performance ao longo do tempo.
- **Plataforma:** testes automatizados de schema na entrada; filas/API para alto volume.
- **Fairness e política de dados:** revisar uso de atributos sensíveis (ex.: `gender`) com critério de negócio e conformidade; documentar decisão de inclusão ou remoção.
- **Features de domínio:** variáveis derivadas (ex.: relação charge/tenure, flags de pacote) se o time de negócio validar hipóteses.
- **Baseline interpretável:** regressão logística ou modelos lineares como referência de coeficientes, para contraste com XGBoost.

---

## Conclusão

O projeto cumpre o núcleo de um fluxo de ciência de dados aplicado a churn: dados públicos Telco, preparação explícita, **duas famílias de abordagem** (Random Forest e XGBoost) com camada de **balanceamento**, métricas alinhadas ao problema (**recall**), score contínuo via **probabilidade** e demonstração em **Streamlit** (simulador e **predição em lote por upload de CSV** com `churn_pred` e `risk_score`). As principais ressalvas são **validação estatística mais forte** e **trade-off precision/recall operacionalizado**, itens naturalmente priorizados em uma segunda iteração profissional.
