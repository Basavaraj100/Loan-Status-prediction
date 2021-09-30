# Loan-Status-prediction

## Problem Statment

**Goal:** Predicting whether the bank customer is eligible for loan sanction or not.
Loan status predicted by using machine learning classification algorithms. After analyzing different models finally DecisionTreeClassifier fitted to solve this problem. Loan status prediction app is created using streamlit python and deployed.<br>

## Approach

### Cleaning Credit score column
- Credit score column contain 4 digit numbers also, after domain knowledge experty and google search we learnt that credit score should be three digit number only.
- We observe that all four digit numbers ends with 0's so we simply drop the ending zero in four digit numbers

- Credit score before cleaning

![image](https://github.com/Basavaraj100/Loan-Status-prediction/blob/main/images/credit_score_before_cleaning.PNG)

-Credit score after cleaning

![After balance](https://github.com/Basavaraj100/Loan-Status-prediction/blob/main/images/credit_score_after_cleaning.PNG)

### Missing value treatment using statistical tests
- For numerical features we checked that whther mean of numerical features is same for both categories of targte(Loan status)
if there is significant difference we replace the missing values with mean value of numerical features of corresponding category
- Here we used two sample ztest
```python
Ho: credit score for fully paid <= credit score for charged off
Ha: credit score for fully paid > credit score for charged off

def cred_score_imputation(cols):
    loan_status=cols[1]
    cred_score=cols[0]
    if pd.isnull(cred_score):
        if loan_status=='Fully Paid':
            return ful_paid_cred_score.median()
        if loan_status=='Charged Off':
            return charged_off_cred_score.median()
    else:
        return cred_score


df['Credit Score']=df[['Credit Score','Loan Status']].agg(cred_score_imputation,axis=1)
```

![image]()

### Balancing the data
- The target column(Loan status) was not balance , we used SMOTE technique to balance the frequency of categories in targte column
- Credit score before cleaning
- Before SMOTE
![image](https://github.com/Basavaraj100/Loan-Status-prediction/blob/main/images/before_balance.PNG)

- After SMOTE

![image](https://github.com/Basavaraj100/Loan-Status-prediction/blob/main/images/after_balance.PNG)


### model selection and workflow
- We tried with each and every model to fit the data and to get bets accuarcy , and selected the best performer among all

![Performance of each model](https://github.com/Basavaraj100/Loan-Status-prediction/blob/main/images/model_performances.jpg)

### Feature selection
- PermutationImportance technique used to select the best features, as the result 21 features are selected, using these fetures agin we build the model

![Selected features are](https://github.com/Basavaraj100/Loan-Status-prediction/blob/main/images/selected_features.png)

### final model
- XgBoost Classifier is the final model, which yield 85% accuracy with only 21 features among(41) features


## model performance
- Confussion matrix

![image](https://github.com/Basavaraj100/Loan-Status-prediction/blob/main/images/Confussion_matrix.png)

-Classification report

![image](https://github.com/Basavaraj100/Loan-Status-prediction/blob/main/images/classification_report.png)

## Model deployment
- Model deployed in Google cloud platform using streamlit
- the supporting files are

a) Dockerfile
```python
FROM python:3.9
WORKDIR /app
COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt
EXPOSE 8080
COPY . /app

CMD streamlit run --server.port 8080 --server.enableCORS false app.py
```

b) app.yaml
```python
runtime: custom
env: flex

```



## Streamlit app
This app contain following section

1)Home: In this section you will find the description of problem statment and source for dataset

2)EDA: This section agin divided into<br>

    A)Descriptive: Here description such as value counts, shape of the data...etc are described<br>
    B) Plots: Here Important plots used in EDA are mentioned<br>

3)ML: In this section you can predict the loan status by entering the required features in the given fields

App link: https://loan-status-prediction-326315.el.r.appspot.com/

