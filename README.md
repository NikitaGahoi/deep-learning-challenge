# deep-learning-challenge

## Overview of the Analysis:
The nonprofit foundation, Alphabet Soup, sought to enhance its funding selection process by developing a predictive tool. The goal was to create a binary classifier using machine learning and neural networks, predicting the success of applicants if funded by Alphabet Soup. The provided dataset contained metadata about over 34,000 organizations, including columns such as `EIN`, `NAME`, `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`, and the target variable `IS_SUCCESSFUL`.

## Data Preprocessing:
 - **STEP 1 :** Identification columns (EIN and NAME) were removed from the input data because they are neither targets nor features
 - **STEP 2 :** Identified the number of unique values present in each columns. `APPLICATION_TYPE` and `CLASSIFICATION`, had more than 10 unique values, therefore created cutoff point of 500 and 1000 respectively to bin "rare" categorical variables together in a new value ("others")
 - **STEP 3 :** Converted categorical data to numeric with pd.get_dummies, then lastly split into training and tesing datasets
 - **STEP 4 :** Split the preprocessed data into features and target variable:
         -Target Variable (y): `IS_SUCCESSFUL`
         -Feature Variable (X): `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE` , `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`
   This resulted in a data that included 43 feature variables 
 - **STEP 5 :** Split the preprocessed data into training and test set using `train_test_split` from sklearn module. `StandardScaler` was used to scale the training data, and then the testing data

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
After pre-processing the data, correlation matrix and Random forest was performed to look for important features and to remove the less important features that are not adding any value to the model. Code can be found in **Random_Forest-Feature_selection.ipynb**

### More Step in Data Pre-processing:
The column "ASK_AMT" had a 8747 unique values ranging from 5000 to 8597806340. To look for distribution/frequency of these 8747 values, `value_counts()` function was applied. the data turned out be extremely skewed, with 5000 having a unique count of 25398

![image](https://github.com/NikitaGahoi/deep-learning-challenge/assets/136101293/b5b9c1c6-6350-4c7d-adca-71d3b56a2d9e)

Since the values in ASK_AMT has a very huge range, logarithmic tranformation and binning (bin = 15) was performed. A histogram was plotted to have a visual representation of values in ASK_AMT

![image](https://github.com/NikitaGahoi/deep-learning-challenge/assets/136101293/d2499a36-f2f1-45f2-aeab-2cacb1780ed9)

Log_bins were created on the log-tansformed data, and the bin were assigned to the corresponding values in the original dataframe. Given below is the value_counts for each bin:

![image](https://github.com/NikitaGahoi/deep-learning-challenge/assets/136101293/b93a69c7-4992-473e-8f1b-8e0dd257f61a)

The histogram represents that the data is largely skewed, to scale the `ASK_AMT` feature log transformed values were used for further analysis, and these log-tranformed values were then acaled using the `standard-scaler` 


### Correlation Matrix: 
The correlation matrix was calculated for the dataset application_dummies. This matrix represents the pairwise correlations between different features. It is then visualized as a heatmap using the seaborn library. It could be concluded from the correlation matrix that two columns "STATUS" and "SPECIAL_CONSIDERATION" have almost no co-relation with any feature in the matrix. To confirm the conclusion drawn from the correlation matrix and look at the feature importance , random forest model was tested and list of important features were created.

![image](https://github.com/NikitaGahoi/deep-learning-challenge/assets/136101293/0605f9c5-cfd0-4b42-9c31-90518f1f6c0d)

### Random Forest Model:
Application_dummies dataset was split into training and test sets and random forest classifier instance was initiated, the model was trained and fitted on training dataset, and its efficiency was evaluated using confusion matrix. The accuracy obtained by the random forest model was 71%

![image](https://github.com/NikitaGahoi/deep-learning-challenge/assets/136101293/25c45c83-8816-4ffe-b215-c5eb6ecae5a1)

Random Forests in sklearn automatically calculates **feature importance**. All the features based on their importance was sorted and visualized (Top 20 important features). The bar graph below represents that the features like  `ASK_AMT`, `AFFILIATION`, `APPLICATION_TYPE`, `CLASSIFICATION`, `USE_CASE` , `ORGANIZATION` are important features 

![image](https://github.com/NikitaGahoi/deep-learning-challenge/assets/136101293/92e874ca-f770-4b3e-8022-e4ad0907c683)

**Least Important Features:** "STATUS" and "SPECIAL_CONSIDERATION" were among the least important feature, along with that "INCOME_AMT" also showed very less importance

![image](https://github.com/NikitaGahoi/deep-learning-challenge/assets/136101293/263fb09f-56ca-4715-a679-c9ff315894cf)







