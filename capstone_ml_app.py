# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 00:00:53 2021

@author: HP
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier
import pickle




# -----------------reading dat--------------------------
# os.chdir(r'C:\Users\HP\Desktop\Data_science\streamlit_data_science_app_building\streamlit_practices')
# clean_bal_df=pd.read_csv(r'balanced_cleaned_train_data.csv')


# # ----------------important features-----------------------

imp_features = ['Annual Income','Home Ownership_Home Mortgage','Credit Score',
                'Current Loan Amount','Home Ownership_Rent','Home Ownership_Own Home',
                'Term_Short Term','Years in current job_10+ years',
                'Purpose_Debt Consolidation','Monthly Debt',
                'Current Credit Balance','Number of Credit Problems',
                'Maximum Open Credit','Years in current job_2 years',
                'Years in current job_3 years','Number of Open Accounts',
                'Years of Credit History','Years in current job_< 1 year',
                'Months since last delinquent','Years in current job_4 years']

# # splitting data
# x=clean_bal_df[imp_features]
# y=clean_bal_df['Loan Status_Fully Paid']
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

# #  model fitting
# base_learners = [('Decision Tree', DecisionTreeClassifier(min_samples_split=41,min_samples_leaf=63,
#                                                           max_depth=36,criterion='entropy')),
#                   ('Random Forest', RandomForestClassifier(random_state=10,n_estimators=400,min_samples_split=5,
#                                     max_features='auto',max_depth=60,bootstrap=False))]

# stack_model_gdBoost_partial = StackingClassifier(estimators = base_learners, final_estimator = GradientBoostingClassifier(random_state = 8))


# stack_model_gdBoost_partial.fit(x_train, y_train)

# y_pred_imp_feature = stack_model_gdBoost_partial.predict(x_test)









def run_ml_app():
    result=None
    model = pickle.load(open('capstone_project_loan_status.pkl', 'rb'))
    
    # --------home ownership--------
    Home_Ownership_Home_Mortgage=0
    Home_Ownership_Rent=0
    Home_Ownership_Own_Home=0
    home_others=0

    # ---------job experience---------
    yrs_exp_10_plus=0
    yrs_exp_2_years=0
    yrs_exp_3_years=0
    yrs_exp_4_years=0
    yrs_exp_less_1=0
    yrs_exp_others=0

    # -------------purpose-----
    purpose=0

    
    st.subheader('Please Enter the following details to Predict Loan Eligibility Status')
    
    col1,col2=st.beta_columns(2)
    
    with col1:
        # -------------annual income-----------
        annual_income=st.number_input('Select annual income',min_value=50000,max_value=100000000)
        
        # -----------------credit score------------------
        credit_score=st.slider('Select credit score',300,800)
        
        # -------------------term---------------
        term=st.radio('Select term',['Short term','Long term'])
        if term=='Short term':
            term_=1
        else:
            term_=0
            
            
        # ------------purpose-------------
        purpose_rad=st.radio('Select Purpose of loan',['Debt consolidation','Others'])
        if purpose_rad=='Debt consolidation':
            purpose=1
            
        # -----------current credit balance--------
        cur_credit_bal=st.number_input('Select current credit balance',0,10000000)
        
        
        # -----------max open credit-------------
        max_open_cred=st.number_input('Select maximum open credit',0,1000000000)
        
        # ----------years of credit history---------------
        yr_cred_hist=st.slider('Select years of credit history',2,80)
        
        
        
      
        
        
        
        
        
        
        
        
        
    with col2:
    #     # ------------home ownership----------------
        h_m=['Mortagage','Rent','Own Home','Others']
        home_ownership=st.selectbox('Select Home ownership type',h_m)
        if home_ownership=='Mortagage':
            Home_Ownership_Home_Mortgage=1
        elif home_ownership=='Rent':
            Home_Ownership_Rent=1
        elif home_ownership=='Own Home':
            Home_Ownership_Own_Home=1
        else:
            home_others=1
            
            
    #     # --------loan amount------------------
        loan_amount=st.number_input('Select current loan amount',10000,100000000)
        
        
    #     # --------------Select job experience---------------
        job_exp_sel=['less than 1 year','2 years','3 years','4 years','10+ years','others']
        job_exp=st.selectbox('Select job experience',job_exp_sel)
        if job_exp=='less than 1 year':
            yrs_exp_less_1=1
        elif job_exp=='2 years':
            yrs_exp_2_years=1
        elif job_exp=='3 years':
            yrs_exp_3_years=1
        elif job_exp=='4 years':
            yrs_exp_4_years=1
        elif job_exp=='10+ years':
            yrs_exp_10_plus=1
        else:
            yrs_exp_others=1
            
            
    #     # -------------monthly debt-----------
        mon_debt=st.number_input('Select monthly debt',0,10000000)
        
        
    #     # -----------number of credit problems---------
        num_cred_prob=st.slider('Select number of credit problems',0,16)
        
        
    #     # --------number of open accounts----------
        num_open_acc=st.slider('Select number of open accounts',0,80,1)
        
        # --------month since last deliquent--------
        deliquent=st.slider('Select month since last deliquent',0,180,1)
        
    with st.beta_expander('Selected inputs'):   
        inp=[annual_income,Home_Ownership_Home_Mortgage,credit_score,loan_amount,Home_Ownership_Rent,Home_Ownership_Own_Home,
              term_,yrs_exp_10_plus,purpose,mon_debt,cur_credit_bal,num_cred_prob,max_open_cred,
              yrs_exp_2_years,yrs_exp_3_years,num_open_acc,yr_cred_hist,yrs_exp_less_1,deliquent,yrs_exp_4_years] 
        
        d=dict(zip(imp_features,inp))
        st.write(d)
        
    if st.button('Predict Loan status'):
        a=np.array(inp)
        a=a.reshape(1,20)
        result=model.predict(a)[0]
        
        
    if result==0:
        st.error('Sorry, you are not eligible for loan')
    elif result==1:
        st.success('You are eligible for loan')
        
        
            
            
            
            
       
            
        
            
        
        
            
            
            
            
            
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    