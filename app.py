import io

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.figure_factory as ff

# Colunas esperadas no CSV estilo Kaggle Telco (Churn e customerID opcionais)
COLUNAS_TELCO_OBRIGATORIAS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def raw_to_model_matrix(df: pd.DataFrame, colunas_treino: list) -> pd.DataFrame:
    """Espelha o pré-processamento de churn.py: retorna X alinhado às colunas do treino."""
    df = df.copy()
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    colunas_binarias = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in colunas_binarias:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    colunas_categoricas = df.select_dtypes(exclude=["number"]).columns
    df = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True, dtype=int)
    return df.reindex(columns=colunas_treino, fill_value=0)


# ==========================================
# 1. CARREGAMENTO DO MODELO
# ==========================================
@st.cache_resource
def load_model():
    modelo = joblib.load("modelo_churn_xgboost.joblib")
    colunas = joblib.load("colunas_treino.joblib")
    return modelo, colunas


modelo, colunas_treino = load_model()

# ==========================================
# 2. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Previsão de Churn | Portfólio", page_icon="📊", layout="wide")
st.title("📊 Retenção Inteligente: Previsão de Churn")
st.write("Um aplicativo de Machine Learning de ponta a ponta para identificar evasão de clientes.")

# Criando as Abas da aplicação
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔮 Simulador de Risco",
        "📁 Predição em lote (CSV)",
        "📈 Insights do Negócio",
        "🧠 Bastidores da IA (Tech)",
    ]
)

# ==========================================
# ABA 1: O SIMULADOR
# ==========================================
with tab1:
    st.header("Simulador de Cliente")
    st.write("Insira as características do cliente para avaliar o risco de cancelamento em tempo real.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tempo de Casa (meses)", 0, 72, 12)
        monthly_charges = st.number_input("Valor da Fatura Mensal ($)", 15.0, 120.0, 50.0)
        total_charges = tenure * monthly_charges
    
    with col2:
        contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Serviço de Internet", ["DSL", "Fiber optic", "No internet service"])
    
    if st.button("Analisar Risco do Cliente", type="primary"):
        dados_novo_cliente = {
            'tenure': tenure, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
            'Contract': contract, 'InternetService': internet, 'SeniorCitizen': 0,
            'Partner': 'No', 'Dependents': 'No', 'PhoneService': 'Yes', 'MultipleLines': 'No',
            'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No',
            'TechSupport': 'No', 'StreamingTV': 'No', 'StreamingMovies': 'No',
            'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check', 'gender': 'Male'
        }
        
        df_novo = pd.DataFrame([dados_novo_cliente])
        X_in = raw_to_model_matrix(df_novo, colunas_treino)
        if X_in.empty:
            st.error("Não foi possível montar as features para este cliente (dados inválidos).")
        else:
            previsao = modelo.predict(X_in)[0]
            probabilidade = modelo.predict_proba(X_in)[0][1] * 100
            st.divider()
            if previsao == 1:
                st.error(f"🚨 **ALTO RISCO DE CHURN!** Probabilidade: {probabilidade:.1f}%.")
                st.warning("Ação sugerida: Entrar em contato com o cliente com uma oferta de retenção.")
            else:
                st.success(f"✅ **CLIENTE SEGURO.** Probabilidade de cancelamento: {probabilidade:.1f}%.")

# ==========================================
# ABA 2: PREDIÇÃO EM LOTE (CSV)
# ==========================================
with tab2:
    st.header("Upload de clientes (CSV)")
    st.write(
        "Envie um arquivo no formato **Telco Customer Churn** (Kaggle). "
        "Colunas obrigatórias alinhadas ao dataset original; `customerID` e `Churn` são opcionais "
        "(se `Churn` existir, é ignorado na predição)."
    )
    st.caption(
        "Referência: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)"
    )

    arquivo = st.file_uploader("Arquivo CSV", type=["csv"])

    if arquivo is not None:
        try:
            raw = pd.read_csv(arquivo)
        except Exception as e:
            st.error(f"Não foi possível ler o CSV: {e}")
        else:
            faltando = [c for c in COLUNAS_TELCO_OBRIGATORIAS if c not in raw.columns]
            if faltando:
                st.error(
                    "Faltam colunas obrigatórias no CSV: **"
                    + "**, **".join(faltando)
                    + "**. Use o mesmo esquema do dataset Telco."
                )
            else:
                n_antes = len(raw)
                ids_serie = raw["customerID"] if "customerID" in raw.columns else None
                X_mat = raw_to_model_matrix(raw, colunas_treino)
                n_depois = len(X_mat)
                if n_depois == 0:
                    st.warning(
                        "Nenhuma linha válida após limpeza (verifique `TotalCharges` numérico e valores ausentes)."
                    )
                else:
                    if n_antes > n_depois:
                        st.info(
                            f"**{n_antes - n_depois}** linha(s) removida(s) por `TotalCharges` inválido ou ausente "
                            f"(mesma regra do treino). **{n_depois}** linha(s) scoring."
                        )
                    churn_pred = modelo.predict(X_mat)
                    risk_score = modelo.predict_proba(X_mat)[:, 1]
                    out = pd.DataFrame(
                        {
                            "churn_pred": churn_pred.astype(int),
                            "risk_score": risk_score,
                        }
                    )
                    if ids_serie is not None:
                        out.insert(0, "customerID", ids_serie.loc[X_mat.index].astype(str).values)
                    else:
                        out.insert(0, "row_index", X_mat.index.astype(int).values)

                    st.subheader("Resultado")
                    st.dataframe(
                        out.assign(
                            risk_score_pct=(out["risk_score"] * 100).round(2)
                        ),
                        use_container_width=True,
                    )

                    csv_buf = io.StringIO()
                    out.assign(risk_score_pct=(out["risk_score"] * 100).round(2)).to_csv(
                        csv_buf, index=False
                    )
                    st.download_button(
                        label="Baixar resultado (CSV)",
                        data=csv_buf.getvalue().encode("utf-8"),
                        file_name="predicoes_churn.csv",
                        mime="text/csv",
                    )

# ==========================================
# ABA 3: INSIGHTS DO NEGÓCIO
# ==========================================
with tab3:
    st.header("O que faz o cliente cancelar?")
    st.write("Este é o 'Raio-X' do algoritmo XGBoost. As variáveis no topo são as que mais pesam na decisão matemática do modelo.")
    
    importancias = modelo.feature_importances_
    df_importancias = pd.DataFrame({'Variável': colunas_treino, 'Importância': importancias})
    df_importancias = df_importancias.sort_values(by='Importância', ascending=True).tail(10)
    
    fig_importancia = px.bar(
        df_importancias, 
        x='Importância', 
        y='Variável', 
        orientation='h',
        color='Importância',
        color_continuous_scale='Reds'
    )
    fig_importancia.update_layout(showlegend=False, height=500, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_importancia, use_container_width=True)
    
    st.info("""
    **Principais Descobertas de Negócio:**
    * **Contratos Mensais (`Contract_Month-to-month`):** São o principal gatilho de cancelamento. A falta de fidelidade facilita a saída.
    * **Fibra Óptica (`InternetService_Fiber optic`):** Clientes com este serviço cancelam mais, indicando possíveis problemas de qualidade ou preço frente à concorrência.
    * **Tempo de Casa (`tenure`):** Quanto mais tempo o cliente passa na base, menor a chance de ele sair.
    """)

# ==========================================
# ABA 4: BASTIDORES DA IA (Com Texto e Matrizes)
# ==========================================
with tab4:
    st.header("Engenharia do Modelo e Tomada de Decisão")
    st.write("Este projeto foi construído focando na **métrica de Recall**, garantindo que a empresa minimize o número de Falsos Negativos (clientes que cancelam sem o modelo perceber).")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("1. O Desafio dos Dados Desbalanceados")
        st.write("Em bancos de dados de evasão, há sempre muito mais clientes que ficam do que clientes que saem. Um modelo tradicional (como Random Forest simples) alcançou apenas **46% de Recall** na identificação de cancelamentos.")
        st.markdown("**A Solução (SMOTE):**")
        st.write("Apliquei a técnica de *Synthetic Minority Over-sampling Technique* (SMOTE) apenas nos dados de treino. Ao gerar dados sintéticos da classe minoritária, forçamos a IA a aprender os padrões de evasão, subindo nossa capacidade de detecção estrutural.")
        
    with colB:
        st.subheader("2. A Escolha do XGBoost")
        st.write("Para maximizar o retorno financeiro da operação, a arquitetura final escolheu o **XGBoost (Extreme Gradient Boosting)**.")
        st.markdown("**O Resultado:**")
        st.write("Combinando SMOTE e XGBoost, o modelo saltou de 46% para **64% de Recall** na classe de cancelamento. Em um cenário real de negócios, isso significa prever e ter a chance de salvar dezenas de clientes a mais todos os meses, justificando tecnicamente o investimento em IA.")

    st.divider()
    
    st.subheader("A Evolução do Algoritmo na Prática")
    st.write("Observe nas matrizes de confusão abaixo como a capacidade de detectar clientes em risco (quadrante inferior direito) aumentou significativamente conforme aplicamos as técnicas explicadas acima.")

    # Função auxiliar para criar os heatmaps
    def plot_matriz(z, title):
        x = ['Fica (Previsto)', 'Cancela (Previsto)']
        y = ['Fica (Real)', 'Cancela (Real)']
        fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
        fig.update_layout(title_text=title, title_x=0.5, height=350, margin=dict(l=0, r=0, t=50, b=0))
        return fig

    # Matrizes lado a lado
    col1, col2, col3 = st.columns(3)
    
    with col1:
        z_rf = [[932, 101], [203, 171]]
        st.plotly_chart(plot_matriz(z_rf, "1. Random Forest Base"), use_container_width=True)
        st.caption("Apenas **46%** dos cancelamentos detectados (171 clientes). O modelo sofreu com o desbalanceamento.")
        
    with col2:
        z_smote = [[868, 165], [164, 210]]
        st.plotly_chart(plot_matriz(z_smote, "2. RF + SMOTE"), use_container_width=True)
        st.caption("A detecção subiu para **56%** (210 clientes). O balanceamento sintético fez o modelo reconhecer melhor a evasão.")
        
    with col3:
        z_xgb = [[826, 207], [135, 239]]
        st.plotly_chart(plot_matriz(z_xgb, "3. XGBoost + SMOTE (Final)"), use_container_width=True)
        st.caption("A detecção saltou para **64%** (239 clientes salvos). O algoritmo Gradient Boosting maximizou a eficiência.")