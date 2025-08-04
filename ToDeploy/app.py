import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and imputer
model = joblib.load('risk_model.pkl')
imputer = joblib.load('imputer.pkl')

st.set_page_config(page_title="Phishing Risk Predictor", layout="centered")

st.title("🔐 Social Media Phishing Risk Predictor")
st.write("Upload your data or enter details below to assess your phishing risk level.")

# Option to upload a file
uploaded_file = st.file_uploader("📄 Upload Excel/CSV file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df_numeric = df.select_dtypes(include='number')
        X_input = imputer.transform(df_numeric)
        predictions = model.predict(X_input)

        df['Predicted Risk Level'] = predictions
        st.success("✅ Predictions completed!")
        st.write(df)

        st.download_button("📥 Download Results", data=df.to_csv(index=False), file_name="risk_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

else:
    st.subheader("📝 Manual Input (for single user)")
    
    # Example input fields (replace with your actual columns)
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    time_spent = st.number_input("Average Time Spent on Social Media (hrs/day)", min_value=0.0, max_value=24.0, value=2.0)
    posts_per_day = st.number_input("Posts per Day", min_value=0, max_value=100, value=3)
    public_profile = st.selectbox("Is your profile public?", ["Yes", "No"])
    
    # Convert to numeric (assuming model uses 0/1 for Yes/No)
    public_profile_num = 1 if public_profile == "Yes" else 0

    # Put into DataFrame
    input_df = pd.DataFrame([[age, time_spent, posts_per_day, public_profile_num]],
                            columns=['Age', 'Time Spent', 'Posts Per Day', 'Public Profile'])

    # Predict
    if st.button("🔍 Predict Risk"):
        X_input = imputer.transform(input_df)
        prediction = model.predict(X_input)[0]
        st.success(f"🎯 Predicted Risk Level: **{prediction}**")
