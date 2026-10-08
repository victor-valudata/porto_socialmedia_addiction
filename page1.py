import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

st.title("Student Social Media Addiction")
st.header("Exploratory Data Analysis")
st.text("This dataset contains survey-based information about students’ social media usage habits and levels of addiction. It includes demographic details such as age, gender, and academic level, along with behavioral variables related to time spent on social media, preferred platforms, frequency of use, and purpose of engagement. The dataset also records indicators of addiction such as difficulty controlling usage, impact on sleep patterns, concentration in studies, emotional dependency, and feelings of anxiety or restlessness when not using social media. Each record represents an individual student’s response, providing a detailed view of how social media use varies across different student groups.")
st.page_link("https://www.kaggle.com/datasets/zahranusratt/student-social-media-addiction-analysis-dataset/code", 
             label="Click here to access the original dataset from Kaggle")

df = pd.read_csv('Students Social Media Addiction.csv')

st.subheader("Raw Data")
st.dataframe(df)

st.subheader("Demography")
tab_gender, tab_age, tab_academic, tab_relationship, tab_platform, tab_country = st.tabs(
    ["Gender", "Age", "Academic", "Relationship", "Platform", "Country"])

with tab_gender:
    df_gender = df.groupby('Gender').size().reset_index(name='GenderCount')
    st.bar_chart(df_gender, x='Gender', y='GenderCount', color=['#0061A4'])

with tab_age:
    df_age = df.groupby('Age').size().reset_index(name='AgeCount')
    st.bar_chart(df_age, x='Age', y='AgeCount', color=['#291871'])

with tab_academic:
    df_academic = df.groupby('Academic_Level').size().reset_index(name='AcademicCount')
    st.bar_chart(df_academic, x='Academic_Level', y='AcademicCount', color=['#D82435'])

with tab_relationship:
    df_relationship = df.groupby('Relationship_Status').size().reset_index(name='RelationshipCount')
    st.bar_chart(df_relationship, x='Relationship_Status', y='RelationshipCount', color=['#F26920'])

with tab_platform:
    top4platform = df["Most_Used_Platform"].value_counts().nlargest(4).index
    df["Top_Most_Used_Platform"] = df["Most_Used_Platform"].where(
        df["Most_Used_Platform"].isin(top4platform), "Others"
    )
    df_platform = df.groupby('Top_Most_Used_Platform').size().reset_index(name='PlatformCount')
    st.bar_chart(df_platform, x='Top_Most_Used_Platform', y='PlatformCount', color=['#FAE609'])

with tab_country:
    top9country = df["Country"].value_counts().nlargest(9).index
    df["Top_Country"] = df["Country"].where(
        df["Country"].isin(top9country), "Others"
    )
    df_country = df.groupby('Top_Country').size().reset_index(name='CountryCount')
    st.bar_chart(df_country, x='Top_Country', y='CountryCount', color=['#00924C'])

st.subheader("Target Distribution")
st.text("The selected target from the dataset is Addicted Score. The distribution and feature correlation are as follow.")
df_addictedscore = df.groupby('Addicted_Score').size().reset_index(name='AddictedScodeCount')
st.bar_chart(df_addictedscore, x='Addicted_Score', y='AddictedScodeCount', color=['#291871'])
st.caption("Figure 1.7. Target (Addicted Score) Distribution. Addicted Score is in integer ranges from 2 to 9.")

st.subheader("Feature Correlation")
df_rel = df.copy()

# Encode categorical columns
label_encoders = {}
categorical_cols = df_rel.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if col != 'Addicted_Score': # Ensure target is handled separately if needed
        le = LabelEncoder()
        df_rel[col] = le.fit_transform(df_rel[col].astype(str))
        label_encoders[col] = le

# If target variable is text/categorical, encode it
if df_rel['Addicted_Score'].dtype == 'object':
    target_le = LabelEncoder()
    df_rel['Addicted_Score'] = target_le.fit_transform(df_rel['Addicted_Score'].astype(str))

features_for_corr = df_rel.drop(columns=['Student_ID', 'Mental_Health_Score'], errors='ignore')
correlation_matrix = features_for_corr.corr()

# Extract correlations with the target variable, sorted
target_corr = correlation_matrix['Addicted_Score'].drop('Addicted_Score').sort_values(ascending=False)
target_corr_pos = []
target_corr_negs = []
for corr in target_corr:
    if corr < 0:
        target_corr_pos.append(0)
        target_corr_negs.append(corr)
    else:
        target_corr_pos.append(corr)
        target_corr_negs.append(0)
df_targetcorr = pd.DataFrame({
    'Positively Relate' : target_corr_pos,
    'Negatively Relate' : target_corr_negs,
    'Features' : target_corr.index.tolist(),
})
st.bar_chart(df_targetcorr, x='Features', y=['Positively Relate','Negatively Relate'] , y_label='', horizontal=True, sort=False, color=['blue','red'])
st.caption("Figure 1.8. Correlation between features and target are shown. Longer bar shows higher correlation while shorter bar shows lower correlation. Negative value indicates negative correlation.")




