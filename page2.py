import streamlit as st
import pandas as pd

st.title("Student Social Media Addiction")
st.header("Machine Learning")
st.subheader("Preprocessing Stage")
st.subheader("Duplicates")
st.text("Finding duplicates with the following script")
st.code("df.duplicated().sum()")
st.text("The script result is 0. No further action is required")
st.subheader("Missing Values")
st.text("Finding missing values with the following script")
st.code("df.isnull().sum()")
st.text("The script result is 0. No further action is required")
st.subheader("Non-Numerical Columns")
st.text("Finding categorical columns with the following script")
st.code("categorical_cols = df.select_dtypes(include=['object']).columns")
st.text("The script result shows some categorical columns. Those columns are 'Gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 'Affects_Academic_Performance', and 'Relationship_Status'")
st.text("LabelEncoder are applied to those columns with the following script")
st.code("label_encoders = {}\n" \
"for col in categorical_cols:\n" \
"\tle = LabelEncoder()\n" \
"\tdf[col] = le.fit_transform(df[col].astype(str))\n" \
"\tlabel_encoders[col] = le")
st.subheader("Miscellaneous")
st.text("Based on Figure 1.7., the Addicted_Score 2 has only 1 member. Since it is heavily imbalanced and makes problem in further train/test split. In order to create better model, the data is removed.")
st.text("Based on Table 1.1. there are some columns need to be removed. Student_ID column has only unique value so this column is ommited. Mental_Health_Score column is considered as target so this column is ommited.")
st.subheader("Training Stage")
st.subheader("Creating Feature and Target")
st.text("Creating Feature and Target is done with the following script")
st.code("X = df_filtered.drop(columns=['Addicted_Score'], errors='ignore')\n" \
"y = df_filtered['Addicted_Score']")
st.subheader("Splitting Testing and Target")
st.text("Splitting testing and target is done with the following script")
st.code("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)")
st.text("Parameter test_size=0.2 is set will split training and testing in 80:20 ratio. Parameter random state=42 is set for replicability purpose. Parameter stratify=y is set to maintain training-testing data proportion mainly due to imbalanced nature of the dataset.")
st.subheader("Training")
st.text("Model is trained with Random Forest method")
st.code("rf_model = RandomForestClassifier(random_state=42, n_estimators=100)')\n" \
"rf_model.fit(X_train, y_train)")
st.text("Parameter random state=42 is set for replicability purpose. Parameter n_estimators=100 is basically default value from scikit-learn library.")
st.subheader("Evaluation Stage")
st.subheader("Testing")
st.text("Testing is held with the following script")
st.code("y_pred = rf_model.predict(X_test)")
st.subheader("Evaluation")
st.text("Evaluation is held with the following script")
st.code("classification_report(y_test, y_pred)")
st.text("While confusion matrix is constructed with the following script")
st.code("confusion_matrix(y_test, y_pred)")
df_evaluation = pd.DataFrame(
    {
        "precision": [1.00, 0.94, 0.96, 0.80, 1.00, 1.00, 1.00, None, 0.96, 0.97],
        "recall": [0.67, 0.94, 0.96, 1.00, 0.95, 1.00, 1.00, None, 0.93, 0.96],
        "f1-score": [0.80, 0.94, 0.96, 0.89, 0.98, 1.00, 1.00, 0.96, 0.94, 0.96],
        "support": [3, 17, 27, 12, 42, 29, 11, 141, 141, 141],
    },
    index=["3", "4", "5", "6", "7", "8", "9", "accuracy", "macro avg", "weighted avg"]
)
st.text("The confusion matrix is shown in Figure 2.1.")
st.image("confusion_matrix.png")
st.caption('Figure 2.1. Confusion matrix from the model testing. The correct predictions are shown in the diagonal boxes from upper left to lower right.')
st.text("Evaluation report is presented in Table 2.1.")
st.caption('Table 2.1. Evaluation Report of the Model')
st.write(df_evaluation)
st.text('From Figure 2.1. there are only small number of missing prediction. Good sign for the model performance. Table 2.1. gives more solid evaluation. Accuracy of 96%, F1-Score (macro) of 94%, Precision (macro) of 96%, and Recall (macro) of 93%.')
st.text('The evaluation shows that training a random forest with the student social media addiction data may result a very good model. High accuracy along with high F1-Score on imbalanced dataset shows that the model predicts accurately across the class with little bias.')
st.text('The model get helped from highly related features as shown in Figure 1.8. Random Forest also helps in combatting considerably imbalanced dataset.')
st.text('While in overall the model evaluation is very good, there are some things to considered')
st.text('1. The dataset consists of 704 rows distributed unevenly across 7 classes when fed upon machine learning stage. This condition makes some classes have only a small amount of data. The example is class 3 that only tested with 3 data. When a model tested with such small number of data, the result may lead to overfitting.')
st.text('2. Class 2 of the dataset is omitted due to insufficient data. To give more comprehensive model, dataset that has sufficient data for every class is advised.')
