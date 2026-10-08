import streamlit as st

pg = st.navigation([
    st.Page("page1.py", title="Exploratory Data Analysis"),
    st.Page("page2.py", title="Machine Learning"),
    st.Page("page3.py", title="Playground"),
])
pg.run()
# import pandas as pd

# st.title("Student Social Media Addiction")
# st.text("https://www.kaggle.com/datasets/zahranusratt/student-social-media-addiction-analysis-dataset/code")
# st.text("This dataset contains survey-based information about students’ social media usage habits and levels of addiction. It includes demographic details such as age, gender, and academic level, along with behavioral variables related to time spent on social media, preferred platforms, frequency of use, and purpose of engagement. The dataset also records indicators of addiction such as difficulty controlling usage, impact on sleep patterns, concentration in studies, emotional dependency, and feelings of anxiety or restlessness when not using social media. Each record represents an individual student’s response, providing a detailed view of how social media use varies across different student groups.")
# st.write("Hello word")
# df = pd.read_csv('Students Social Media Addiction.csv')

# st.dataframe(df)

# tab_gender, tab_age, tab_academic, tab_relationship, tab_platform = st.tabs(
#     ["Gender", "Age", "Academic", "Relationship", "Platform"])

# with tab_gender:
#     df_gender = df.groupby('Gender').size().reset_index(name='GenderCount')
#     st.bar_chart(df_gender, x='Gender', y='GenderCount', color=['#0061A4'])

# with tab_age:
#     df_age = df.groupby('Age').size().reset_index(name='AgeCount')
#     st.bar_chart(df_age, x='Age', y='AgeCount', color=['#291871'])

# with tab_academic:
#     df_academic = df.groupby('Academic_Level').size().reset_index(name='AcademicCount')
#     st.bar_chart(df_academic, x='Academic_Level', y='AcademicCount', color=['#D82435'])

# with tab_relationship:
#     df_relationship = df.groupby('Relationship_Status').size().reset_index(name='RelationshipCount')
#     st.bar_chart(df_relationship, x='Relationship_Status', y='RelationshipCount', color=['#FAE609'])

# with tab_platform:
#     df_platform = df.groupby('Most_Used_Platform').size().reset_index(name='PlatformCount')
#     st.bar_chart(df_platform, x='Most_Used_Platform', y='PlatformCount', color=['#00924C'])

# gender_options = {
#     0: "Male",
#     1: "Female"
# }

# selected_value = st.selectbox(
#     "Gender",
#     options=gender_options.keys(),
#     format_func=lambda x: gender_options[x]
# )

# st.write("Selected value:", selected_value)
# st.write("Selected label:", gender_options[selected_value])