import streamlit as st

pg = st.navigation([
    st.Page("page1.py", title="Exploratory Data Analysis"),
    st.Page("page2.py", title="Machine Learning"),
    st.Page("page3.py", title="Playground"),
])
pg.run()

# git add .
# git commit -m "Updated project"
# git push
