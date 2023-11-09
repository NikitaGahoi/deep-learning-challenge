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

## Compiling, Training, And Evaluating Models:
**ATTEMPT 1**: For the first attempt (HDF5_Files/AlphabetSoupCharity.h5), hyperparameters used were:

![image](https://github.com/NikitaGahoi/deep-learning-challenge/assets/136101293/bedc73a9-113e-4991-be82-31e5d4c61fda)

**Model 1 Evaluation using test data:**
-Accuracy: 72.61%
-Loss: 55.84%

**ATTEMPT 2:** For the second attempt (HDF5_Files/AlphabetSoupCharity_nn_2.h5), hyperparameters used were:

![image](https://github.com/NikitaGahoi/deep-learning-challenge/assets/136101293/5b1fef12-8314-46f6-b0ff-fd3c0a6ecf04)

**Model 2 Evaluation using test data:**
-Accuracy: 72.78%
-Loss: 56.70%

**ATTEMPT 3:** For the second attempt (HDF5_Files/AlphabetSoupCharity_nn_3.h5), hyperparameters used were:

![image](https://github.com/NikitaGahoi/deep-learning-challenge/assets/136101293/d872867f-5f0a-4244-9382-2bd462455e4b)

**Model 3 Evaluation using test data:**
-Accuracy: 72.69%
-Loss: 56.21%

**Conclusion:** Despite multiple attempts, and increasing the nodes and layers, the target predictive accuracy of 75% was not reached. Therefore, I looked into the feature importance to reduce the dimension of the X-variable, by removing the least important features from the data. To achieve this objective, two approches were used "correlation_matrix" and "random forest model" to calculate feature importance.

## Analyzing Important Features:
After 


