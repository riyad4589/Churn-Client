import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "models/churn_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="Telco Churn Prediction",page_icon="📶", layout="wide")
st.title("📉 Système de prédiction de churn client")

tab1, tab2 = st.tabs(["🔮 Prédiction individuelle", "📊 Dashboard simple"])

with tab1:
    st.subheader("Entrer les caractéristiques du client")

    col1, col2, col3 = st.columns(3)

    # ---------------------------
    # COLONNE 1 : numériques
    # ---------------------------
    with col1:
        tenure = st.number_input("Ancienneté (mois)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Facture mensuelle", min_value=0.0, max_value=500.0, value=70.0)
        total_charges = st.number_input("Total payé", min_value=0.0, max_value=10000.0, value=1500.0)
        senior = st.selectbox("SeniorCitizen", ["0", "1"])  # 0 = non, 1 = oui

    # ---------------------------
    # COLONNE 2 : infos client / services
    # ---------------------------
    with col2:
        gender = st.selectbox("Genre", ["Male", "Female"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        phone_service = st.selectbox("PhoneService", ["Yes", "No"])
        multiple_lines = st.selectbox("MultipleLines", ["No phone service", "No", "Yes"])

    # ---------------------------
    # COLONNE 3 : internet / options
    # ---------------------------
    with col3:
        internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("OnlineSecurity", ["No internet service", "No", "Yes"])
        online_backup = st.selectbox("OnlineBackup", ["No internet service", "No", "Yes"])
        device_protection = st.selectbox("DeviceProtection", ["No internet service", "No", "Yes"])
        streaming_tv = st.selectbox("StreamingTV", ["No internet service", "No", "Yes"])
        streaming_movies = st.selectbox("StreamingMovies", ["No internet service", "No", "Yes"])
        tech_support = st.selectbox("TechSupport", ["No internet service", "No", "Yes"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])
        payment_method = st.selectbox(
            "PaymentMethod",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    if st.button("Prédire le risque de churn"):
        # IMPORTANT : noms EXACTS comme dans le CSV
        input_dict = {
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        X_input = pd.DataFrame([input_dict])

        proba = model.predict_proba(X_input)[0, 1]
        pred = model.predict(X_input)[0]

        st.markdown("---")
        st.write(f"**Probabilité de churn : {proba*100:.2f} %**")
        if pred == 1:
            st.error("⚠️ Client à **fort risque** de churn.")
        else:
            st.success("✅ Client à **faible risque** de churn.")


with tab2:
    st.subheader("Dashboard sur le dataset Telco Churn")

    # ---------- Chargement des données ----------
    @st.cache_data
    def load_dataset():
        df = pd.read_csv("telco_churn.csv")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.dropna(subset=["TotalCharges"])
        return df

    df = load_dataset()

    # ---------- Filtres interactifs ----------
    st.markdown("### 🎛️ Filtres")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        contrats = ["All"] + sorted(df["Contract"].unique().tolist())
        filtre_contrat = st.selectbox("Type de contrat", contrats)

    with col_f2:
        internet_services = ["All"] + sorted(df["InternetService"].unique().tolist())
        filtre_internet = st.selectbox("Service Internet", internet_services)

    with col_f3:
        tenure_min, tenure_max = int(df["tenure"].min()), int(df["tenure"].max())
        filtre_tenure = st.slider(
            "Ancienneté (mois)",
            min_value=tenure_min,
            max_value=tenure_max,
            value=(tenure_min, tenure_max),
        )

    # Appliquer les filtres
    df_filtre = df.copy()
    if filtre_contrat != "All":
        df_filtre = df_filtre[df_filtre["Contract"] == filtre_contrat]
    if filtre_internet != "All":
        df_filtre = df_filtre[df_filtre["InternetService"] == filtre_internet]
    df_filtre = df_filtre[
        (df_filtre["tenure"] >= filtre_tenure[0]) &
        (df_filtre["tenure"] <= filtre_tenure[1])
    ]

    # ---------- KPIs sur le segment filtré ----------
    st.markdown("### 📌 Indicateurs globaux (segment filtré)")
    col_a, col_b, col_c, col_d = st.columns(4)

    total_clients = len(df_filtre)
    churn_rate = (df_filtre["Churn"].map({"Yes": 1, "No": 0}).mean()) * 100
    avg_tenure = df_filtre["tenure"].mean()
    avg_monthly = df_filtre["MonthlyCharges"].mean()

    col_a.metric("Nombre de clients", total_clients)
    col_b.metric("Taux de churn (%)", f"{churn_rate:.2f}")
    col_c.metric("Ancienneté moyenne (mois)", f"{avg_tenure:.1f}")
    col_d.metric("Facture mensuelle moyenne", f"{avg_monthly:.1f}")

    # ---------- Répartition churn / non-churn ----------
    st.markdown("### 📊 Répartition churn / non-churn")
    churn_counts = df_filtre["Churn"].value_counts()
    st.bar_chart(churn_counts)

    # ---------- Churn par variable de segmentation ----------
    st.markdown("### 🔎 Taux de churn par variable")

    var_segment = st.selectbox(
        "Variable de segmentation",
        ["Contract", "InternetService", "PaymentMethod", "gender", "SeniorCitizen"]
    )

    churn_by_var = (
        df_filtre.groupby(var_segment)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )
    st.bar_chart(churn_by_var)

    # ---------- Distributions Tenure / MonthlyCharges ----------
    st.markdown("### 📈 Distribution de l'ancienneté et des montants selon le churn")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("Ancienneté par churn")
        st.bar_chart(
            df_filtre.groupby("Churn")["tenure"].mean()
        )

    with col_g2:
        st.markdown("Facture mensuelle par churn")
        st.bar_chart(
            df_filtre.groupby("Churn")["MonthlyCharges"].mean()
        )
