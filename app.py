import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.figure_factory as ff

# ==========================================
# 1. CARREGAMENTO DO MODELO
# ==========================================
@st.cache_resource
def load_model():
    modelo = joblib.load('modelo_churn_xgboost.joblib')
    colunas = joblib.load('colunas_treino.joblib')
    return modelo, colunas

modelo, colunas_treino = load_model()

# ==========================================
# 2. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Previsão de Churn | Portfólio", page_icon="📊", layout="wide")
st.title("📊 Retenção Inteligente: Previsão de Churn")
st.write("Um aplicativo de Machine Learning de ponta a ponta para identificar evasão de clientes.")

# Criando as Abas da aplicação
tab1, tab2, tab3 = st.tabs(["🔮 Simulador de Risco", "📈 Insights do Negócio", "🧠 Bastidores da IA (Tech)"])

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
        df_novo['gender'] = df_novo['gender'].map({'Male': 1, 'Female': 0})
        colunas_binarias = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in colunas_binarias:
            df_novo[col] = df_novo[col].map({'Yes': 1, 'No': 0})
        
        colunas_categoricas = df_novo.select_dtypes(exclude=['number']).columns
        df_novo = pd.get_dummies(df_novo, columns=colunas_categoricas, drop_first=True, dtype=int)
        df_novo = df_novo.reindex(columns=colunas_treino, fill_value=0)
        
        previsao = modelo.predict(df_novo)[0]
        probabilidade = modelo.predict_proba(df_novo)[0][1] * 100
        
        st.divider()
        if previsao == 1:
            st.error(f"🚨 **ALTO RISCO DE CHURN!** Probabilidade: {probabilidade:.1f}%.")
            st.warning("Ação sugerida: Entrar em contato com o cliente com uma oferta de retenção.")
        else:
            st.success(f"✅ **CLIENTE SEGURO.** Probabilidade de cancelamento: {probabilidade:.1f}%.")

# ==========================================
# ABA 2: INSIGHTS DO NEGÓCIO
# ==========================================
with tab2:
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
# ABA 3: BASTIDORES DA IA (Com Texto e Matrizes)
# ==========================================
with tab3:
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