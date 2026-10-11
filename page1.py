import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

st.title("Student Social Media Addiction")
st.header("Exploratory Data Analysis")
st.text("The dataset is about students addiction to social media taken from Kaggle. The dataset composed from 705 student contributors from many countries. The dataset includes demography of its contributors, social media behavior and the contributor score. The score composed of mental health score and addiction score.")
st.page_link("https://www.kaggle.com/datasets/zahranusratt/student-social-media-addiction-analysis-dataset/code", 
             label="Click here to access the original dataset from Kaggle")
st.text("The dataset is interesting since currently social media has taken the world. Despite the content that may bring harm, the nature of social media is also worrying since it lures people to consume the content more and more. Such behavior that may lead to addiction. Social media addiction may have big impact in health, performance and overall daily life. Students as younger generation not only prone for such addiction but also arguably receive more damage since there are still years ahead of them to live on.")
st.text("In this EDA section, the dataset is explored to show the overall contributor demography, their social media behavior and basic analysis of the addiction.")

df = pd.read_csv('Students Social Media Addiction.csv')

st.subheader("Raw Data")

st.text("The has 705 rows and 13 columns. The columns are Student ID, Age, Gender, Academic Level, Country, Average Daily Usage Hours, Most Used Platform, Affects on Academic Performance, Sleep Hours per Night, Mental Health Score, Relationship Status, Conflicts Over Social Media, and Addicted Score.")

with st.expander("Browse the raw data"):
    st.dataframe(df)

st.subheader("Demography")
tab_gender, tab_age, tab_academic, tab_relationship, tab_country = st.tabs(
    ["Gender", "Age", "Academic", "Relationship", "Country"])

with tab_gender:
    df_gender = df.groupby('Gender').size().reset_index(name='GenderCount')
    st.bar_chart(df_gender, x='Gender', y='GenderCount', color=['#0061A4'])
    st.caption('Figure 1.1. Gender distribution. Distribution of male and female is balanced.')

with tab_age:
    df_age = df.groupby('Age').size().reset_index(name='AgeCount')
    st.bar_chart(df_age, x='Age', y='AgeCount', color=['#291871'])
    st.caption('Figure 1.2. Age distribution. Contributors age are within 19-24 with the majority ages between 19-22yo.')

with tab_academic:
    df_academic = df.groupby('Academic_Level').size().reset_index(name='AcademicCount')
    st.bar_chart(df_academic, x='Academic_Level', y='AcademicCount', color=['#D82435'])
    st.caption('Figure 1.3. Academic level distribution. Most contributors are undergraduate or graduate student. Only few contributors come from highschool student.')

with tab_relationship:
    df_relationship = df.groupby('Relationship_Status').size().reset_index(name='RelationshipCount')
    st.bar_chart(df_relationship, x='Relationship_Status', y='RelationshipCount', color=['#F26920'])
    st.caption('Figure 1.4. Relationshop distribution. Most contributors are either in relationship or single. Contributors with complicated relationship only in small number.')

with tab_country:
    top9country = df["Country"].value_counts().nlargest(9).index
    df["Top_Country"] = df["Country"].where(
        df["Country"].isin(top9country), "Others"
    )
    df_country = df.groupby('Top_Country').size().reset_index(name='CountryCount')
    st.bar_chart(df_country, x='Top_Country', y='CountryCount', color=['#00924C'])
    st.caption('Figure 1.6. Country distribution. The contributors come from 110 countries. For better clarity, only 9 countries with top contributors are shown. The other countries are grouped as Other.')

st.text("Demography wise, nothing special about the contributors. In general, age and academic level are as expected from the target contributors. The relationship status of the contributors are also common. Perhaps the only interesting take from the demography is the large number of origin country of the contributors at 110. However, the distribution is cosiderably balanced. While the correlation of the origin country to the target can be seen in the Feature Correlation section.")

st.subheader("Social Media Behavior")
tab_platform, tab_daily_hour = st.tabs(["Platform", "Daily Usage"])

with tab_platform:
    top4platform = df["Most_Used_Platform"].value_counts().nlargest(4).index
    df["Top_Most_Used_Platform"] = df["Most_Used_Platform"].where(
        df["Most_Used_Platform"].isin(top4platform), "Others"
        )
    df_platform = df.groupby('Top_Most_Used_Platform').size().reset_index(name='PlatformCount')
    df_platform = df_platform.sort_values(
        by="Top_Most_Used_Platform",
        key=lambda x: x.eq("Others")
        ).reset_index(drop=True)
    st.bar_chart(df_platform, x='Top_Most_Used_Platform', y='PlatformCount', color=['#FAE609'], sort=False)
    st.caption('Figure 1.5. Platform distribution. There are actually 12 platforms mostly used by the contributors. To create better visualization, only top 4 are shown. The other platforms are combined as Other.')

with tab_daily_hour:
    bins = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, np.inf]

    labels = [
        "1.5-2.5",
        "2.5-3.5",
        "3.5-4.5",
        "4.5-5.5",
        "5.5-6.5",
        "6.5-7.5",
        "7.5-.."
    ]

    df["Daily_Hours"] = pd.cut(
        df["Avg_Daily_Usage_Hours"],
        bins=bins,
        labels=labels,
        right=False
    )
    df_dailyhour = df.groupby('Daily_Hours').size().reset_index(name='DailyHoursCount')
    st.bar_chart(df_dailyhour, x='Daily_Hours', y='DailyHoursCount', color=['#FAE609'], sort=False)
    st.caption('Figure 1.5. Platform distribution. There are actually 12 platforms mostly used by the contributors. To create better visualization, only top 4 are shown. The other platforms are combined as Other.')
    # st.write(df_dailyhour)


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

features_for_corr = df_rel.drop(columns=['Student_ID', 'Mental_Health_Score', 'Daily_Hours', 'Top_Most_Used_Platform', "Top_Country"], errors='ignore')
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




