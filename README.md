# deep-learning-challenge

## Overview of the Analysis:
The nonprofit foundation, Alphabet Soup, sought to enhance its funding selection process by developing a predictive tool. The goal was to create a binary classifier using machine learning and neural networks, predicting the success of applicants if funded by Alphabet Soup. The provided dataset contained metadata about over 34,000 organizations, including columns such as EIN, NAME, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, and the target variable IS_SUCCESSFUL.

## Data Preprocessing:
 - **STEP 1 :** Identification columns (EIN and NAME) were removed from the input data because they are neither targets nor features
 - **STEP 2 :** Identified the number of unique values present in each columns. APPLICATION_TYPE and CLASSIFICATION, had more than 10 unique values, therefore created cutoff point of 500 and 1000 respectively to bin "rare" categorical variables together in a new value ("others")
 - **STEP 3 :** Converted categorical data to numeric with pd.get_dummies, then lastly split into training and tesing datasets
 - **STEP 4 :** Split the preprocessed data into features and target variable:
         -Target Variable (y): "IS_SUCCESSFUL"
         -Feature Variable (X): "APPLICATION_TYPE", "AFFILIATION", "CLASSIFICATION", "USE_CASE", "ORGANIZATION", "STATUS", "INCOME_AMT", "SPECIAL_CONSIDERATIONS", "ASK_AMT"
   This resulted in a data that included 43 feature variables 
 - **STEP 5 :** Split the preprocessed data into training and test set using "train_test_split" from sklearn module

## COMPILING, TRAINING, AND EVALUATING THE MODEL:
**ATTEMPT 1**
The first attempt (Models/AlphabetSoupCharity1.h5) resulted in an accuracy score of 72.6%. This means that 72.6% of the model’s predicted values align with the dataset’s true values.

The hyperparameters used were:

layers = 2
layer 1 = 80 neurons : activation function = "relu"
layer2 = 30 neurons : activation function = "tanh"
epochs = 100
![image](https://github.com/NikitaGahoi/deep-learning-challenge/assets/136101293/37e1add1-4060-4723-a0e6-76fae886cfd1)

Model 1 Evaluation using test data:
Accuracy: 72.61%
Loss: 55.84%

**ATTEMPT 2**
The first attempt (Models/AlphabetSoupCharity1.h5) resulted in an accuracy score of 72.8%. This was the highest accuracy score of the three models. This means that 72.8% of the model’s predicted values align with the dataset’s true values.

The hyperparameters used were:
layers = 2
layer1 = 9 neurons : activation function = ‘relu’
layer2 = 18 neurons : activation function = ‘relu'
epochs = 100



**ATTEMPT 3**
The first attempt (Models/AlphabetSoupCharity1.h5) resulted in an accuracy score of 72.8%. This was the highest accuracy score of the three models. This means that 72.8% of the model’s predicted values align with the dataset’s true values.

The hyperparameters used were:

layers = 2
layer1 = 9 neurons : activation function = ‘relu’
layer2 = 18 neurons : activation function = ‘relu'
epochs = 100
Model 1 Accuracy Plot

