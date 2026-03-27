import kagglehub
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib


path = kagglehub.dataset_download("blastchar/telco-customer-churn") 
print("Path to dataset files:", path) 

file_path = path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

df.info()

# Converter espaços vazios em NaN e depois para numérico
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.dropna(subset=['TotalCharges'], inplace=True)

if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

# 1. Transformar o nosso Target (Churn) em 1 e 0 explicitamente
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})


# 2. Transformar as colunas puramente binárias (Yes / No)
colunas_binarias = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in colunas_binarias:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Transformar o Gênero (Gender)

# 3. One-Hot Encoding para categorias múltiplas
# Usamos exclude=['number'] para pegar tudo que NÃO for número (ou seja, os textos restantes)
# Isso evita qualquer conflito ou warning de versão do Pandas
colunas_categoricas = df.select_dtypes(exclude=['number']).columns

# Aplicamos o get_dummies (dtype=int garante que teremos 1 e 0 em vez de True/False)
df = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True, dtype=int)

# 4. Checar como ficou a nossa base de dados
print(f"Novo formato da base: {df.shape[0]} linhas e {df.shape[1]} colunas")
df.head()




# 1. Calcula a correlação de todas as colunas com 'Churn'
correlacoes = df.corr()['Churn'].sort_values(ascending=False)

# 2. Remove o próprio 'Churn' (pois a correlação dele com ele mesmo é 1.0)
correlacoes = correlacoes.drop('Churn')

# 3. Configuração do gráfico
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# 4. Criação do gráfico de barras (Vermelho para correlação positiva, Azul para negativa)
cores = ['#e74c3c' if x > 0 else '#3498db' for x in correlacoes]
ax = sns.barplot(x=correlacoes.values, y=correlacoes.index, palette=cores)

# 5. Ajustes visuais para o portfólio
plt.title('Fatores de Risco para o Cancelamento (Churn)', fontsize=18, pad=20)
plt.xlabel('Coeficiente de Correlação (Pearson)', fontsize=12)
plt.ylabel('Variáveis do Cliente', fontsize=12)
plt.axvline(x=0, color='black', linewidth=1.5)

plt.tight_layout()
plt.show()

# 6. Exibindo os Top 5 de cada lado no terminal
print("🔥 TOP 5 Fatores que AUMENTAM o risco de cancelamento:")
print(correlacoes.head(5))

print("\n🛡️ TOP 5 Fatores que DIMINUEM o risco de cancelamento:")
print(correlacoes.tail(5))


# 1. Separar as variáveis explicativas (X) da variável alvo (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# 2. Dividir em Treino (80%) e Teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dados de Treino: {X_train.shape[0]} linhas")
print(f"Dados de Teste: {X_test.shape[0]} linhas\n")

# 3. Criar e treinar o modelo de Machine Learning
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
modelo_rf.fit(X_train, y_train)

# 4. Fazer as previsões usando os dados de teste (que o modelo nunca viu)
previsoes = modelo_rf.predict(X_test)

# 5. Avaliar o desempenho
print("📊 Relatório de Classificação:\n")
print(classification_report(y_test, previsoes))

# 6. Matriz de Confusão visual
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, previsoes), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Cancelou (0)', 'Cancelou (1)'], 
            yticklabels=['Não Cancelou (0)', 'Cancelou (1)'])
plt.title('Matriz de Confusão do Modelo')
plt.ylabel('Realidade (O que de fato aconteceu)')
plt.xlabel('Previsão do Modelo')
plt.show()


# 1. Instanciando e aplicando o SMOTE APENAS nos dados de treino (X_train, y_train)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Mostrando a mágica do SMOTE acontecendo
print("📊 Distribuição ORIGINAL do Treino:")
print(y_train.value_counts())
print("\n⚖️ Distribuição APÓS SMOTE (Treino Balanceado):")
print(y_train_smote.value_counts())
print("-" * 50)

# 2. Treinando o modelo novamente com os novos dados
# Como os dados já estão 50/50, não precisamos mais do 'class_weight'
modelo_rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf_smote.fit(X_train_smote, y_train_smote)

# 3. Fazendo previsões nos dados de Teste (que NUNCA viram o SMOTE)
previsoes_smote = modelo_rf_smote.predict(X_test)

# 4. Avaliando o novo desempenho
print("\n🚀 NOVO Relatório de Classificação (Com SMOTE):\n")
print(classification_report(y_test, previsoes_smote))

# 5. Nova Matriz de Confusão
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, previsoes_smote), annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Cancelou (0)', 'Cancelou (1)'], 
            yticklabels=['Não Cancelou (0)', 'Cancelou (1)'])
plt.title('Matriz de Confusão - Modelo com SMOTE')
plt.ylabel('Realidade (O que de fato aconteceu)')
plt.xlabel('Previsão do Modelo')
plt.show()



# 1. Instanciar o modelo XGBoost
# O eval_metric e use_label_encoder são apenas para evitar warnings chatos da biblioteca
modelo_xgb = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 2. Treinar o modelo com os dados balanceados do SMOTE
modelo_xgb.fit(X_train_smote, y_train_smote)

# 3. Fazer as previsões
previsoes_xgb = modelo_xgb.predict(X_test)

# 4. Avaliar o resultado do XGBoost
print("🔥 Relatório de Classificação (XGBoost + SMOTE):\n")
print(classification_report(y_test, previsoes_xgb))

# 5. Matriz de Confusão visual
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, previsoes_xgb), annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Não Cancelou (0)', 'Cancelou (1)'], 
            yticklabels=['Não Cancelou (0)', 'Cancelou (1)'])
plt.title('Matriz de Confusão - XGBoost')
plt.ylabel('Realidade (O que de fato aconteceu)')
plt.xlabel('Previsão do Modelo')
plt.show()


# Extrair a importância das variáveis do modelo XGBoost
importancias = modelo_xgb.feature_importances_

# Criar um DataFrame para facilitar a visualização
df_importancias = pd.DataFrame({
    'Variavel': X_train_smote.columns,
    'Importancia': importancias
})

# Pegar as 10 variáveis mais importantes
df_importancias = df_importancias.sort_values(by='Importancia', ascending=False).head(10)

# Criar o gráfico
plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Variavel', data=df_importancias, palette='viridis')
plt.title('Top 10 Fatores de Decisão do XGBoost', fontsize=16)
plt.xlabel('Peso de Importância no Modelo', fontsize=12)
plt.ylabel('Variável', fontsize=12)
plt.tight_layout()
plt.show()



# 1. Salvar o modelo XGBoost treinado
joblib.dump(modelo_xgb, 'modelo_churn_xgboost.joblib')

# 2. Salvar o nome das colunas do X_train_smote
# Isso garante que o Streamlit saiba exatamente quais colunas o modelo espera receber
colunas_treino = X_train_smote.columns.tolist()
joblib.dump(colunas_treino, 'colunas_treino.joblib')

print("✅ Modelo e estrutura de colunas salvos com sucesso!")
