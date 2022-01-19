from multiprocessing import dummy
import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Penguin Prediction App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Featurs')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

    penguins_raw = pd.read_csv('penguins_cleaned.csv')
    penguins= penguins_raw.drop(columns=['species'])
    df=pd.concat([input_df,penguins], axis=0)

encode = ['sex','island']
for col in encode:
    dummy= pd.get_dummies(df[col], prefix=col)
    df=pd.concat([df,dummy], axis=1)
    del df[col]

df= df[:1]
st.subheader('User Input Header')
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awiting for csv uploaded. currently using example input parameters (show below)')
    st.write(df)

load_clf = pickle.load(open('penguins_clf.pkl',"rb"))
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Predction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(str(penguins_species[prediction]))
st.subheader('prediction Probability')
st.write(prediction_proba)



hide_st_style = """ 
<style>
#MainMenu {visibility: hidden}
footer {visibility: hidden}
header {visibility: hidden}
</style>
""" 
st.markdown(hide_st_style , unsafe_allow_html=True)
