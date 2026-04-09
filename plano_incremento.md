# Plano de incremento

Documento único de fechamento do repositório frente ao edital (substitui os rascunhos `plano_encremedo.md` e `plano_encremento.md`).

**Objetivo:** alinhar o projeto ao máximo ao desafio, priorizando o que a banca vê primeiro e o que bloqueia a entrega.

**Tempo total alvo (referência):** ~30 minutos por leva — ajuste conforme o prazo real.

---

## Status do repositório — entregas realizadas

| Bloco | Situação |
|--------|----------|
| Prioridade 1 — relatório + `requirements.txt` | Feito |
| Prioridade 2 — CSV em lote + README | Feito |
| Prioridade 3 — notebook EDA + pipeline descrito no relatório | Feito |
| App — caminho absoluto para `.joblib`, erros claros | Feito |

---

## Prioridade 0 — Já coberto (não gastar tempo)

- `PLANO.md` — leitura do problema, etapas, critérios.
- Modelagem com **≥2 abordagens** (RF baseline, RF+SMOTE, XGB+SMOTE).
- Score contínuo: **`predict_proba`** no app como probabilidade de churn.
- README com instruções de execução.

---

## Prioridade 1 — impacto alto

| # | Ação | Por quê |
|---|------|--------|
| 1 | **`RELATORIO_FINAL.md`:** o que funcionou, o que não, o que faria depois. | Entrega do desafio + base para perguntas ao vivo. |
| 2 | **`requirements.txt`** completo para `churn.py` + app. | Instalação reprodutível ao clonar o repo. |

---

## Prioridade 2 — aderência ao enunciado

| # | Ação | Por quê |
|---|------|--------|
| 3 | **Streamlit — upload de CSV:** lote com `churn_pred` e `risk_score`, download. | Item “plataforma de inferência” do desafio. |
| 4 | **README:** `PLANO.md`, `RELATORIO_FINAL.md`, notebook, fluxo `churn.py` → app. | Repositório autoexplicativo. |

---

## Prioridade 3 — refinamentos

| # | Ação | Por quê |
|---|------|--------|
| 5 | **Notebook EDA** (`notebooks/01_eda.ipynb`). | Item “EDA em notebook” do desafio. |
| 6 | **Pipeline de features** no `RELATORIO_FINAL.md` (ou README). | Item “pipeline / justificativa” sem obrigar `sklearn.Pipeline`. |

---

## Ordem sugerida de execução

1. `RELATORIO_FINAL.md`  
2. `requirements.txt`  
3. Upload CSV no `app.py`  
4. README  
5. Notebook EDA ou reforço do texto de pipeline no relatório  

---

## Se o tempo acabar no meio

- **Mínimo aceitável:** `RELATORIO_FINAL.md` + `requirements.txt` + README.  
- **Plano B na apresentação:** se faltar lote CSV, mostrar simulador + mesma lógica de `predict_proba` (hoje o CSV já está implementado).

---

## Checklist antes de enviar / apresentar

- [x] `PLANO.md` no repositório  
- [x] `plano_incremento.md` (este arquivo) no repositório  
- [x] `RELATORIO_FINAL.md` preenchido  
- [x] `streamlit run app.py` sobe com `.joblib` na pasta do `app.py` (após `python churn.py`)  
- [x] README aponta o notebook `notebooks/01_eda.ipynb`  
- [x] Artefatos: `python churn.py` gera `modelo_churn_xgboost.joblib` e `colunas_treino.joblib`

---

## Próximos passos (fora do código)

1. **Apresentação ao vivo:** ensaiar narrativa (negócio → EDA → pipeline → métricas/recall → score → demo Streamlit com CSV de exemplo).  
2. **Se a banca pedir `sklearn.Pipeline` formal:** hoje a lógica está centralizada em `telco_preprocess.py`; dá para embrulhar em `ColumnTransformer` + `Pipeline` sem mudar regras (refatoração opcional).  
3. **Opcional:** versionar ou não os `.joblib` no Git (hoje não estão no `.gitignore`; CSV já é ignorado).
