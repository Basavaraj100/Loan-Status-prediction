# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 23:46:37 2021

@author: HP
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
import sklearn
from capstone_ml_app import run_ml_app
from capstone_eda_app import run_eda_app
import streamlit.components.v1 as stc 
from PIL import Image


   
        
home_page_info='''#### About Problem Statement 
Using different information from the customer who wants loan, deciding whether he is eligible for loan or not
#### About data
Dataset contains different features including categorical and numerical
#### Data source- https://www.kaggle.com/zaurbegiev/my-dataset'''
    
        
def main():
    img1 = Image.open('loan_app_logo.jpg')
    img1 = img1.resize((300,200))
    st.image(img1,use_column_width=False)
    
    
    menu=['Home','EDA','ML']
    choise=st.sidebar.selectbox('Menu',menu)
    
    
    if choise=='Home':
        st.header('Welcome to Home')
        
        st.markdown(home_page_info)
    elif choise=='EDA':
        run_eda_app()
        
        
    elif choise=='ML':
        run_ml_app()
        
  
    
    
if __name__=='__main__':
    main()