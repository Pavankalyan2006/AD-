import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

st.title('Model Deployment: Decision Tree Classifier')

st.sidebar.header('User Input Parameters')

def user_input_features():
    CLMSEX = st.sidebar.selectbox('Gender', ('1', '0'))
    CLMINSUR = st.sidebar.selectbox('Insurance', ('1', '0'))
    SEATBELT = st.sidebar.selectbox('Seatbelt', ('1', '0'))
    CLMAGE = st.sidebar.number_input("Insert the Age")
    LOSS = st.sidebar.number_input("Insert Loss")

    data = {
        'CLMSEX': CLMSEX,
        'CLMINSUR': CLMINSUR,
        'SEATBELT': SEATBELT,
        'CLMAGE': CLMAGE,
        'LOSS': LOSS
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)

claimants = pd.read_csv('claimants.csv')
claimants.drop(['CASENUM'], inplace=True, axis=1)
claimants = claimants.dropna()

x = claimants.iloc[:, [1,2,3,4,5]]
y = claimants.iloc[:, 0]
clf = DecisionTreeClassifier()
clf.fit(x, y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')   
st.write('Probability of Yes:', prediction_proba[0][1])
st.write('Probability of No:', prediction)
