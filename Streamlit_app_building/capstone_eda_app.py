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
# import plotly.express as px


# Uncleaned data
df_uncleaned=pd.read_csv(r'credit_train.csv')

df_cleaned=pd.read_csv(r'credit_train_cleaned.csv')




def run_eda_app():
    st.subheader('Exploratory Data Analysis')
    opt=['Descriptive','Plots']
    submenu=st.sidebar.selectbox('Submenu',opt)
    
    
    if submenu=='Descriptive':
        with st.beta_expander('Shape of data'):
            st.write(df_uncleaned.shape)
        with st.beta_expander('Understanding data types'):
            st.dataframe(df_uncleaned.dtypes)
            
            
        with st.beta_expander('Five point summary for numerical features'):
            st.dataframe(df_uncleaned.describe(include=np.number))
            
        with st.beta_expander('Categorical feature description'):
            st.dataframe(df_uncleaned.describe(include=object))
            
        with st.beta_expander('Loan status distribution'):
            st.dataframe(df_uncleaned['Loan Status'].value_counts())
            
        
    
    if submenu=='Plots':
        col1,col2=st.beta_columns(2)
        
        with col1:
            # --------credit score---------
            with st.beta_expander('Credit score before cleaning'):
                fig=plt.figure()
                sns.stripplot(data=df_uncleaned,x='Credit Score',y='Loan Status')
                st.pyplot(fig)
                
            # ------------annual income---------
            with st.beta_expander('Annual income stripplot'):
                fig2=plt.figure()
                sns.stripplot(data=df_uncleaned,y='Annual Income',x='Loan Status')
                st.pyplot(fig2)
                
            # ------------loan status---------
            with st.beta_expander('Loan status countplot'):
                fig4=plt.figure()
                sns.countplot(y=df_uncleaned['Loan Status'])
                st.pyplot(fig4)
                
                st.dataframe(df_uncleaned['Loan Status'].value_counts())
                
                
                # --------term--------
            with st.beta_expander('countplot of term '):
                fig5=plt.figure()
                sns.countplot(y=df_uncleaned['Term'])
                st.pyplot(fig5)
                
                
            # -------years in current job---------
            with st.beta_expander('countplot Years in current job '):
                fig6=plt.figure()
                sns.countplot(y=df_uncleaned['Years in current job'])
                st.pyplot(fig6)
                
                
            # ---------home ownership---------
            with st.beta_expander('countplot Home ownership '):
                fig7=plt.figure()
                sns.countplot(y=df_uncleaned['Home Ownership'])
                st.pyplot(fig7)
            
            
            
            
            # ------purpose--------------
            with st.beta_expander('countplot Purpose '):
                fig8=plt.figure()
                sns.countplot(y=df_uncleaned['Purpose'])
                st.pyplot(fig8)
            
            
                
            
            
        with col2:    
            # ----------credit score-----------
            with st.beta_expander('Credit score after cleaning'):
                fig1=plt.figure()
                sns.stripplot(data=df_cleaned,x='Credit Score',y='Loan Status')
                st.pyplot(fig1)
            
            # ----------annual income----------
            with st.beta_expander('Annual income boxplot'):
                fig3=plt.figure()
                sns.boxplot(data=df_uncleaned,x='Annual Income',)
                plt.xscale('log')
                st.pyplot(fig3)
                # ------loan status--------
            with st.beta_expander('Loan status value counts percentage'):
                st.dataframe(df_uncleaned['Loan Status'].value_counts(normalize=True)*100)
                
                
                
            # ------------term--------
            with st.beta_expander('Value counts of term'):
                st.dataframe(df_uncleaned['Term'].value_counts())
                
                
            # ------years in current job---------
            with st.beta_expander('Value counts of years in current job'):
                st.dataframe(df_uncleaned['Years in current job'].value_counts())
                
                
                
            # --------home ownership------------
            with st.beta_expander('Value counts of Home Ownership'):
                st.dataframe(df_uncleaned['Home Ownership'].value_counts())
            
            
            
            
            # ------purpose-----------
            with st.beta_expander('Value counts of Purpose'):
                st.dataframe(df_uncleaned['Purpose'].value_counts())
                
                
        with st.beta_expander('Correlation plot'):
            cov_mat=round(df_uncleaned.corr(),2)
            fig10=plt.figure(figsize=(10,10))
            sns.heatmap(cov_mat,annot=True)
            st.pyplot(fig10)
            
            
                
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        