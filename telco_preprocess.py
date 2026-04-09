"""
Pré-processamento Telco (Kaggle) — mesma lógica em treino (`churn.py`) e inferência (`app.py`).
"""

from __future__ import annotations

import pandas as pd

# Colunas mínimas para CSV estilo Kaggle (Churn e customerID opcionais na inferência)
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


def preprocess_telco_raw(df: pd.DataFrame, *, inferencia: bool = False) -> pd.DataFrame:
    """
    Limpa e codifica o DataFrame Telco.

    - inferencia=False (treino): exige `Churn` Yes/No; mapeia para 0/1 e mantém a coluna.
    - inferencia=True: remove `Churn` se existir; retorna só features numéricas + dummies.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if inferencia:
        if "Churn" in df.columns:
            df = df.drop(columns=["Churn"])
    else:
        if "Churn" not in df.columns:
            raise ValueError("Para treino, o DataFrame deve conter a coluna 'Churn' (Yes/No).")
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    colunas_binarias = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in colunas_binarias:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    colunas_categoricas = df.select_dtypes(exclude=["number"]).columns
    df = pd.get_dummies(df, columns=colunas_categoricas, drop_first=True, dtype=int)
    return df


def align_to_training_columns(X: pd.DataFrame, colunas_treino: list) -> pd.DataFrame:
    """Garante a mesma ordem e conjunto de colunas do artefato salvo no treino."""
    return X.reindex(columns=colunas_treino, fill_value=0)


def features_for_model(df: pd.DataFrame, colunas_treino: list) -> pd.DataFrame:
    """Pipeline lógico de inferência: pré-processa e alinha às colunas do `.joblib`."""
    X = preprocess_telco_raw(df, inferencia=True)
    return align_to_training_columns(X, colunas_treino)
