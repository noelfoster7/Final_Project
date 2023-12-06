#!/usr/bin/env python
# coding: utf-8

# # Final Project
# ## Noel Foster
# ### 12/04/2023

# <hr>

# #### 1. Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[2]:


import pandas as pd
s = pd.read_csv('/Users/noelfoster/Downloads/Gtown Python work/Final Project/social_media_usage.csv')
print(s.shape)


# <hr>

# #### 2. Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[3]:


import pandas as pd
import numpy as np

def clean_sm(x):
    return np.where(x == 1, 1, 0)

#Toy df
data = {
    'A': [1, 4, 9],
    'B': [2, 1, 1]
}
toy_df = pd.DataFrame(data)

cleaned_df = toy_df.apply(clean_sm)

print("Original DataFrame:")
print(toy_df)
print("\nCleaned DataFrame:")
print(cleaned_df)


# <hr>

# #### 3. Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[41]:


import pandas as pd
import numpy as np

def clean_sm(x):
    return np.where(x == 2, 1, 0)

#Make sm_li variable
s['sm_li'] = clean_sm(s['web1h'])

#Select columns
selected_columns = ['income', 'educ2', 'par', 'marital', 'gender', 'age', 'sm_li']

#Drop na's
ss = s[selected_columns].dropna()

#Exploratory analysis
summary_stats = ss.describe()
feature_target_relationships = ss.groupby('sm_li').mean()

print(summary_stats)
print(feature_target_relationships)


# <hr>

# #### 4. Create a target vector (y) and feature set (X)

# In[42]:


y = ss['sm_li']
X = ss.drop(columns=['sm_li'])

print("Target vector (y):")
print(y)

print("\nFeature set (X):")
print(X)


# <hr>

# #### 5. Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[43]:


from sklearn.model_selection import train_test_split

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Explanation of objects:
# - X_train: Set of features for training the model (80% of original dataset) and used to teach the model to predict the target variable.
# - X_test: Set of features for testing the model (20% of original dataset). Used to test the model and evaluate model performance.
# - y_train: Contains 80% of data, this is the target variable for the training model corresponding to X_train. 
# - y_test: Contains 20% of data, this is the target variable for testing the model corresponding to X_test

# <hr>

# #### 6. Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[44]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced')

# Fit model w/ training data
model.fit(X_train, y_train)


# <hr>

# #### 7. Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[45]:


from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(X_test)

#Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# The confusion matrix that our model is 65.45% accurate. We correctly predicted a linkedin user 117 times but also predicted a person was linkedin user who actually wasn't 24 times (~83% correct). We also correctly predicted 80 users who did not use linkedin, but also predicted 80 users did, who actually didn't (50%). This shows the model is very good at predicting people who do use linkedin, but not very good at predicting people who don't use linkedin. 

# <hr>

# #### 8. Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[47]:


import numpy as np

cm = np.array([[80, 24],
               [80, 117]])

#Confusion matrix DF
confusion_df = pd.DataFrame(cm, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
print(confusion_df)


# <hr>

# #### 9. Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# - Precision = True Positive / (True Positive + False Positive)
# - Recall = True Positive / (True Positive + False Negative)
# - F1 Score = 2 x ((Precision x Recall) / (Precision + Recall))

# In[48]:


# Calculating precision, recall, and F1 score
true_positive = 117
false_positive = 24
false_negative = 80

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * (precision * recall) / (precision + recall)

# Displaying the calculated metrics
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")


from sklearn.metrics import classification_report

# Create a classification report
report = classification_report(y_test, y_pred) 
print(report)


# <hr>

# #### 10. Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# In[49]:


import numpy as np

# Person 1's characteristics with six features
person_1 = np.array([8, 7, 0, 1, 42, 0]).reshape(1, -1)  # Adding the sixth feature with value 0

# Person 2's characteristics with six features
person_2 = np.array([8, 7, 0, 1, 82, 0]).reshape(1, -1)  # Adding the sixth feature with value 0

# Use the trained model to predict the probability of using LinkedIn for both persons
prob_person_1 = model.predict_proba(person_1)[:, 1]  # Probability of using LinkedIn for person 1
prob_person_2 = model.predict_proba(person_2)[:, 1]  # Probability of using LinkedIn for person 2

print(f"Probability Person 1 uses LinkedIn: {prob_person_1[0]}")
print(f"Probability Person 2 uses LinkedIn: {prob_person_2[0]}")


# In[50]:


# Person 3's 40, low income, low education prob to be married
person_3 = np.array([2, 2, 0, 1, 40, 0]).reshape(1, -1)  # Adding the sixth feature with value 0

# Person 4's 40, high income, high education prob to be married
person_4 = np.array([7, 7, 0, 1, 40, 0]).reshape(1, -1)  # Adding the sixth feature with value 0

# Use the trained model to predict the probability of using LinkedIn for both persons
prob_person_3 = model.predict_proba(person_3)[:, 1]  # Probability of using LinkedIn for person 1
prob_person_4 = model.predict_proba(person_4)[:, 1]  # Probability of using LinkedIn for person 2

print(f"Probability Person 3 is married: {prob_person_3[0]}")
print(f"Probability Person 4 is married: {prob_person_4[0]}")


# <hr>
