import streamlit as st
import joblib
import pandas as pd

st.title("Student Social Media Addiction")
st.header("Playground")
df = pd.read_csv('Students Social Media Addiction.csv')
data_length = len(df)
def predict_addiction(user_input_dict):
    # 1. Load saved assets
    loaded_model = joblib.load('rf_addiction_model.joblib')
    loaded_encoders = joblib.load('label_encoders.joblib')
    model_features = joblib.load('model_features.joblib')

    # 2. Convert input to a DataFrame
    input_df = pd.DataFrame([user_input_dict])

    # 3. Encode categorical columns using our saved encoders
    if not loaded_encoders:
        raise ValueError("The loaded 'label_encoders.joblib' is empty! Please re-run the pre-processing and saving cells first.")

    for col, encoder in loaded_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col].astype(str))
            except ValueError:
                # Fallback to unseen categories
                input_df[col] = 0

    # 4. Ensure correct column alignment and ordering
    input_df = input_df[model_features]

    # 5. Run prediction
    prediction = loaded_model.predict(input_df)
    return prediction[0]

st.write("---")
st.subheader("Available dataset")
st.text("With the available dataset, you may not only predict the addiction score but find out does it match the ground truth!")
slide_number = st.slider(
    "Use slider to select the Student ID", min_value=1, max_value=data_length
)
# st.write("You're scheduled for:", appointment)
number = st.number_input("Or write the Student ID", value=slide_number, min_value=1, max_value=data_length)
st.write("Selected Student ID:", number)
if st.button('Click here for prediction'):
    ground_truth_addicted_score = df.iloc[number-1]["Addicted_Score"]
    if ground_truth_addicted_score == 2:
        st.text("Sorry. The selected data can't be processed. Please select another data!")
    else:
        selected_data = {
            'Age': df.iloc[number-1]["Age"],
            'Gender': df.iloc[number-1]["Gender"],
            'Academic_Level': df.iloc[number-1]["Academic_Level"],
            'Country': df.iloc[number-1]["Country"],
            'Avg_Daily_Usage_Hours': df.iloc[number-1]["Avg_Daily_Usage_Hours"],
            'Most_Used_Platform': df.iloc[number-1]["Most_Used_Platform"],
            'Affects_Academic_Performance': df.iloc[number-1]["Affects_Academic_Performance"],
            'Sleep_Hours_Per_Night': df.iloc[number-1]["Sleep_Hours_Per_Night"],
            'Relationship_Status': df.iloc[number-1]["Relationship_Status"],
            'Conflicts_Over_Social_Media': df.iloc[number-1]["Conflicts_Over_Social_Media"],
        }
        st.text("Selected Data")
        st.write(selected_data)
        try:
            predicted_addicted_score = predict_addiction(selected_data)
            st.text("Addicted Score")
            st.write("Predicted", predicted_addicted_score)
            st.write("Truth", ground_truth_addicted_score)
            if predicted_addicted_score == ground_truth_addicted_score:
                st.text('Match')
            else:
                st.text('Not Match')
        except Exception as e:
            st.text(f"Error during prediction: {e}")

st.write("---")
st.subheader("Custom Data")
st.text("With custom data, create any data that you want and find out the addiction score!")
age = st.selectbox("Age", sorted(df["Age"].unique().tolist()))
gender = st.selectbox("Gender", sorted(df["Gender"].unique().tolist(), reverse=True))
academic_level = st.selectbox("Academic Level", sorted(df["Academic_Level"].unique().tolist()))
country = st.selectbox("Country", sorted(df["Country"].unique().tolist()))
avg_daily_usage_hours = st.selectbox("Avg Daily Usage (Hours)", sorted(df["Avg_Daily_Usage_Hours"].unique().tolist()))
most_used_platform = st.selectbox("Most Used Platform", sorted(df["Most_Used_Platform"].unique().tolist()))
affects_academic_performance = st.selectbox("Affects Academic Performance", sorted(df["Affects_Academic_Performance"].unique().tolist(), reverse=True))
sleep_hours_per_night = st.selectbox("Sleep Hours Per Night", sorted(df["Sleep_Hours_Per_Night"].unique().tolist()))
relationship_status = st.selectbox("Relationship Status", sorted(df["Relationship_Status"].unique().tolist(), reverse=True))
conflicts_over_social_media = st.selectbox("Conflicts Over Social Media", sorted(df["Conflicts_Over_Social_Media"].unique().tolist()))
if st.button('Click here for prediction (custom data)'):
    custom_data = {
        'Age': age,
        'Gender': gender,
        'Academic_Level': academic_level,
        'Country': country,
        'Avg_Daily_Usage_Hours': avg_daily_usage_hours,
        'Most_Used_Platform': most_used_platform,
        'Affects_Academic_Performance': affects_academic_performance,
        'Sleep_Hours_Per_Night': sleep_hours_per_night,
        'Relationship_Status': relationship_status,
        'Conflicts_Over_Social_Media': conflicts_over_social_media,
    }
    try:
        predicted_addicted_score = predict_addiction(custom_data)
        st.write("Your addicted score", predicted_addicted_score)
    except Exception as e:
        st.text(f"Error during prediction: {e}")

