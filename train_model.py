import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_PATH = "telco_churn.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    # TotalCharges est souvent " " -> convertir en numérique
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # On supprime uniquement les lignes où TotalCharges est NaN
    df = df.dropna(subset=["TotalCharges"])
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(["Churn", "customerID"], axis=1)
    return X, y


def build_pipeline(X):
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )

    pipe = Pipeline(steps=[
        ("preproc", preproc),
        ("model", clf)
    ])
    return pipe

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(X)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(pipe, "models/churn_model.pkl")
    print("✅ Modèle sauvegardé dans models/churn_model.pkl")

if __name__ == "__main__":
    main()
