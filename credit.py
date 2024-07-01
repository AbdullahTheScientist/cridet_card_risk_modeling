import streamlit as st
import pandas as pd
import pickle

st.title('Credit Card Risk Prediction')
st.sidebar.header('Enter Detail')

def user_input_features():
    age = st.sidebar.slider('Age', 18, 59, 32)
    gender = st.sidebar.selectbox('Gender', ['male', 'female'])
    job = st.sidebar.selectbox('Job', ['0', '1', '2', '3'])
    housing = st.sidebar.selectbox('Housing', ['own', 'free', 'rent'])
    saving_accounts = st.sidebar.selectbox('Saving Account', ['little', 'moderate', 'quite rich', 'rich'])
    checking_account = st.sidebar.selectbox('Checking Account', ['little', 'moderate', 'rich'])
    credit_amount = st.sidebar.slider('Credit Amount', 200, 1000, 18424)
    duration = st.sidebar.slider('Duration in months', 1, 20, 80)
    purpose = st.sidebar.selectbox('Purpose', ['car', 'radio/TV', 'furniture/equipment', 'business',
                                               'education', 'repairs', 'domestic appliances',
                                               'vacation/others'])
    data = {
        'Age': age,
        'Sex': gender,
        'Job': job,
        'Housing': housing,
        'Saving accounts': saving_accounts,
        'Checking account': checking_account,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Purpose': purpose
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

if st.button('Predict'):
    # Encoding categorical features
    sex_dummies = pd.get_dummies(input_df['Sex'], drop_first=True, prefix='Sex')
    housing_dummies = pd.get_dummies(input_df['Housing'], drop_first=True, prefix='Housing')
    saving_accounts_dummies = pd.get_dummies(input_df['Saving accounts'], drop_first=True, prefix='Saving_accounts')
    checking_account_dummies = pd.get_dummies(input_df['Checking account'], drop_first=True, prefix='Checking_account')
    purpose_dummies = pd.get_dummies(input_df['Purpose'], drop_first=True, prefix='Purpose')

    input_df = input_df.drop(['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose'], axis=1)
    
    input_df = input_df.merge(sex_dummies, left_index=True, right_index=True)
    input_df = input_df.merge(housing_dummies, left_index=True, right_index=True)
    input_df = input_df.merge(saving_accounts_dummies, left_index=True, right_index=True)
    input_df = input_df.merge(checking_account_dummies, left_index=True, right_index=True)
    input_df = input_df.merge(purpose_dummies, left_index=True, right_index=True)
    
    # Convert 'Job' column to numeric
    input_df['Job'] = input_df['Job'].astype(int)
    
    st.write(input_df)
    
    # Load your model
    model = pickle.load(open('model.pkl', 'rb'))
    
    # Ensure all columns required by the model are present
    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[model.feature_names_in_]
    
    # Prediction
    st.subheader('Prediction')
    prediction = model.predict(input_df)
    if prediction == True:
        st.write('Bad')
    else:
        st.write('Good')

# Note: Uncomment the model loading and prediction code once you have your model.pkl file
