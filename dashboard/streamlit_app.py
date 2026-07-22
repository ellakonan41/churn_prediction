import streamlit as st
import requests
import plotly.graph_objects as go
from src.explanability import get_feature_importance

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📞", layout="wide")
st.title("Customer Churn Prediction")
st.write("This app predicts the likelihood of a customer churning based on their profile and usage data.")


st.header("Customer Information")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Senior Citizen",
    [0,1]
)

partner = st.selectbox(
    "Partner",
    ["Yes","No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes","No"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=100,
    value=12
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes","No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes","No","No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL","Fiber optic","No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes","No","No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes","No","No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes","No","No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes","No","No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes","No","No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes","No","No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month","One year","Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes","No"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    value=840.0
)


predict = st.button("Predict Churn")

if predict:
    st.success("Prediction started")

customer = {
    "gender": gender,
    "SeniorCitizen": senior,
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
    "TotalCharges": total_charges
}

if predict:

    response = requests.post(
        "http://api:8000/predict",
        json = customer
    )

    result = response.json()

    probability = result["churn_probability"]
    prediction = result["churn_prediction"]


    st.subheader("Résultat de l'analyse")


    if prediction == 1:

        st.error(
            f"⚠️ customer with high churn risk\n\n"
            f"churn probability : {probability:.1%}"
        )

    else:

        st.success(
            f"✅ customer with low churn risk\n\n"
            f"churn probability : {probability:.1%}"
        )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value= probability*100,
            title={
                "text":"Risque de churn (%)"
            },
            gauge={
                "axis":{
                    "range":[0,100]
                }
            }
        )
    )


    st.plotly_chart(fig)

    explanation = get_feature_importance(customer)

    st.subheader("🔍 Pourquoi cette prédiction ?")

    fig2 = go.Figure(
        go.Bar(
            x=explanation["shap_value"],
            y=explanation["feature"],
            orientation="h",
            marker_color=explanation["shap_value"].apply(
                lambda x: "red" if x > 0 else "green"
            )
        )
    )

    fig2.update_yaxes(autorange="reversed")

    fig2.update_layout(
        title="Top 5 des facteurs influençant la prédiction",
        xaxis_title="Contribution SHAP",
        yaxis_title="Variables",
        legend_title="Impact",
        height=400
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Interprétation")

    for _, row in explanation.iterrows():

        if row["shap_value"] > 0:

            st.write(
                f"🔴 **{row['feature']}** augmente le risque de churn "
                f"(+{row['shap_value']:.2f})"
            )

        else:

            st.write(
                f"🟢 **{row['feature']}** réduit le risque de churn "
                f"({row['shap_value']:.2f})"
            )