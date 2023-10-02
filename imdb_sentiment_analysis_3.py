#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score
from collections import Counter
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[2]:


#Load the IMDb movie review dataset
df = pd.read_csv(r"C:\Users\ashu1\OneDrive\Desktop\IMDB Dataset3.csv")
df.head(10)


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df['label'].unique()


# In[6]:


sentiment_count = Counter(df['label'])
print(sentiment_count)


# In[7]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Preprocessing steps
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove the word "br"
    text = re.sub(r'\bbr\b', '', text)

    # Tokenize the text (split into individual words)
    words = text.split()

      # Remove stopwords and lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords and len(word) > 1]

    # Join the processed words back into a single string
    processed_text = ' '.join(words)

    # Replace multiple spaces with a single space
    processed_text = re.sub(r'\s+', ' ', processed_text)

    return processed_text

# Apply preprocessing to the text column
df['text'] = df['text'].apply(preprocess_text)

# Apply preprocessing to the "text" column and convert the "label" column to numeric values
df['text'] = df['text'].apply(preprocess_text)
df['label'] = df['label'].map({'positive': 1, 'negative': 0})


# In[8]:


df.head(10)


# In[9]:


sns.countplot(x='label', data=df)
plt.title('Distribution of Sentiments')
plt.show()


# In[10]:


df['text_length'] = df['text'].apply(len)

sns.histplot(x='text_length', data=df)
plt.title('Distribution of Text Lengths')
plt.show()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the text lengths for each review and store them in a list
text_lengths = df['text'].apply(len)

# Create a boxplot using Seaborn
sns.boxplot(x=text_lengths)

# Set the title of the boxplot
plt.title('Boxplot of Text Lengths')

# Display the plot
plt.show()


# In[12]:


xmin = 0
xmax = 1500

### This line creates a boolean mask by checking if the length of each
##text in the 'text' column of the DataFrame df falls within the specified range.
#It uses the apply method along with the len function to calculate the length of each text.
mask = (df['text'].apply(len) >= xmin) & (df['text'].apply(len) <= xmax)

df = df.loc[mask, ['text', 'label']].copy()


df['text_length'] = df['text'].apply(len)

sns.histplot(x='text_length', data=df)
plt.title('Distribution of Text Lengths')
plt.show()


# In[13]:


df.shape


# In[14]:


df.head()


# In[15]:


sns.countplot(x='label', data=df)
plt.title('Distribution of Sentiments')
plt.show()


# In[16]:


sns.histplot(x='text_length', data=df, hue='label', multiple='stack')
plt.title('Distribution of Text Lengths by Sentiment')
plt.show()


# In[17]:


#The lambda function splits each text into words using the split() method,
#..and then calculates the length of the resulting list of words.
word_count = df['text'].apply(lambda x: len(x.split()))
sns.histplot(word_count)
plt.title('Distribution of Word Counts')
plt.show()


# In[18]:


df['word_count'] = word_count
df.groupby('word_count')['label'].mean().plot.hist()
plt.ylabel('word_count')
plt.xlabel('Label')
plt.title('Sentiment vs word_count')
plt.savefig('Sentiment vs word_count')
plt.show()


# In[19]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Filter positive and negative reviews
positive_reviews = df[df['label'] == 1]['text']
negative_reviews = df[df['label'] == 0]['text']

# Create word cloud for positive reviews
positive_wordcloud = WordCloud(width=800, height=400).generate(' '.join(positive_reviews))

# Create word cloud for negative reviews
negative_wordcloud = WordCloud(width=800, height=400).generate(' '.join(negative_reviews))

# Display the word cloud for positive reviews
plt.figure(figsize=(14, 12))
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Word Cloud - Positive Reviews')
plt.axis('off')
plt.show()

# Display the word cloud for negative reviews
plt.figure(figsize=(14, 12))
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Word Cloud - Negative Reviews')
plt.axis('off')
plt.show()


# In[20]:


#                                      🔴   TF-IDF   🔴
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Extract features using TF-IDF vectorization
tfidf = TfidfVectorizer()  # You can further customize the vectorizer if needed
X_train = tfidf.fit_transform(train_df['text']) #It converts the text data into a sparse matrix representation, 
                                                 #...where each row corresponds to a document (review)
                                                 #...and each column corresponds to a unique word in the corpus.

X_test = tfidf.transform(test_df['text'])


# Create the target labels
y_train = train_df['label'] # extracts the target labels from the 'label' column of the training set train_df
                              #...and assigns them to the variable y_train
y_test = test_df['label']


# In[21]:


#                                     🔴 LOGISTIC REGRESSION🔴
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression                                      
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)


# Evaluate the model
accuracy_lr= accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy_lr)
print(f"Classification Report:\n{classification_report}")


# In[22]:


from sklearn.linear_model import LogisticRegression

# Assuming you have already trained the model and called it 'model'

# Get the intercept (bias) of the model
intercept = model.intercept_

# Get the coefficients of the model
coefficients = model.coef_

# Get the unique classes in the target variable
classes = model.classes_

# Print the results
print("Intercept (Bias):", intercept)
print("Coefficients:", coefficients)
print("Classes:", classes)


# # Precision , Recall and F1-score
# Precision: Precision is calculated using the formula: Precision = True Positives / (True Positives + False Positives) Recall: Recall is calculated using the formula: Recall = True Positives / (True Positives + False Negatives)
# 
# To illustrate these formulas, let's consider an example: Suppose we have a binary classification problem where we are predicting whether emails are spam or not.
# 
# True Positives (TP): The model correctly predicts 100 emails as spam. False Positives (FP): The model incorrectly predicts 20 non-spam emails as spam. False Negatives (FN): The model incorrectly predicts 30 spam emails as non-spam. Using these values, we can calculate precision and recall: Precision = 100 / (100 + 20) = 0.833 (83.3%) Recall = 100 / (100 + 30) = 0.769 (76.9%)
# 
# Using these values, we can calculate the F1 score: F1 score = 2 * (0.833 * 0.769) / (0.833 + 0.769) = 0.799 or 79.9 %

# In[22]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#                                     🔴   NAIVE BAYES CLASSIFIER  🔴 
#                                                  
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = naive_bayes.predict(X_test)

# Evaluate the performance
accuracy_nb = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy_nb}")
print(f"Classification Report:\n{classification_report}")


# In[23]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
#                                   🔴  RANDOM FOREST CLASSIFIER  🔴
#                                                
rf = RandomForestClassifier()

# Train the classifier
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred)
classification_repor = classification_report(y_test, y_pred)

print("Accuracy:", accuracy_rf)
print(f"Classification Report:\n{classification_repor}")


# In[24]:


print("Logestic Regression Accuracy:", accuracy_lr)
print("Naive Bayes Accuracy:", accuracy_nb)
print("Random Forest Accuracy:", accuracy_rf)


# In[25]:



#                               🔴  HYPERPARAMETER TUNING  🔴
#                                    LOGISTIC REGRESSION

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
# Define the hyperparameter grid
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Perform grid search cross-validation
grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params1 = grid_search.best_params_
best_model1 = grid_search.best_estimator_

# Make predictions on the testing set using the best model
y_pred = best_model1.predict(X_test)

# Evaluate the best model
accuracy_lrCV = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
print("Best Accuracy:", accuracy_lrCV)
print("Best Hyperparameters:", best_params1)
print(f"Classification Report:\n{classification_report}")


# In[26]:



#                                 🔴   HYPERPARAMETER TUNING   🔴
#                                         NAIVE BAYES

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Define the hyperparameter grid
param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'fit_prior': [True, False]
}

# Perform grid search cross-validation
grid_search = GridSearchCV(estimator=MultinomialNB(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params2 = grid_search.best_params_
best_model2 = grid_search.best_estimator_

# Make predictions on the testing set using the best model
y_pred = best_model2.predict(X_test)

# Evaluate the best model
accuracy_nbCV = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
print("Best Accuracy:", accuracy_nbCV)
print("Best Hyperparameters:", best_params2)
print(f"Classification Report:\n{classification_report}")


# In[50]:


import seaborn as sns
import matplotlib.pyplot as plt

models = ['Logistic Regression','Naive Bayes', 'Random Forest','Logistic Regression after CV','Naive Bayes after CV']
accuracies = [accuracy_lr, accuracy_nb, accuracy_rf,accuracy_lrCV, accuracy_nbCV]

# Set a color palette for the models
colors = sns.color_palette('Set2', n_colors=len(models))

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=accuracies, y=models, palette=colors)

ax.set_title('Model Accuracies TF-IDF', fontsize=16)
ax.set_xlabel('Accuracy', fontsize=14)
ax.set_ylabel('Models', fontsize=14)
ax.set_xlim(0, 1)
ax.tick_params(axis='both', labelsize=12)

# Add the accuracy values on each bar
for i, acc in enumerate(accuracies):
    ax.text(acc+0.01, i, f'{acc:.4f}', va='center', fontsize=12)

plt.show()


# In[28]:


#                                                 🔴   BAG OF WORDS  🔴

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
count_vectorizer = CountVectorizer()  
X_train = count_vectorizer.fit_transform(train_df['text'])
X_test = count_vectorizer.transform(test_df['text'])

y_train = train_df['label']
y_test = test_df['label']


# In[29]:


#                                           🔴 LOGISTIC REGRESSION MODEL🔴
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
                                                       
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_lr2 = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print("Logestic Regression Accuracy:", accuracy_lr2)
print(f"Classification Report:\n{classification_report}")


# In[30]:


#                                              🔴  NAIVE BAYES  🔴

from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB                                                             
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_nb2 = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print("Naive Bayes Accuracy:", accuracy_nb2)
print(f"Classification Report:\n{classification_report}")


# In[31]:


#                                                         🔴   RANDOM FOREST  🔴
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

                                                              
model = RandomForestClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy_rf2 = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
print("Random Forest Accuracy:", accuracy_rf2)
print(f"Classification Report:\n{classification_report}")


# In[32]:


#                               🔴  HYPERPARAMETER TUNING  🔴
#                                    LOGISTIC REGRESSION
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report


# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Extract features using Bag-of-Words vectorization
vectorizer = CountVectorizer()  # You can further customize the vectorizer if needed
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

# Create the target labels
y_train = train_df['label']
y_test = test_df['label']

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'sag', 'saga']
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params3 = grid_search.best_params_
best_model3 = grid_search.best_estimator_

# Make predictions on the testing set
y_pred = best_model3.predict(X_test)

# Evaluate the model
accuracy_lrCV2 = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy_lrCV2)
print("Best Hyperparameters:", best_params3)
print(f"Classification Report:\n{classification_report}")


# In[33]:


#                                 🔴   HYPERPARAMETER TUNING   🔴
#                                          NAIVE BAYES
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report



# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Extract features using Bag-of-Words vectorization
vectorizer = CountVectorizer()  # You can further customize the vectorizer if needed
X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

# Create the target labels
y_train = train_df['label']
y_test = test_df['label']

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'alpha': [0.1, 1, 10]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params4 = grid_search.best_params_
best_model4 = grid_search.best_estimator_

# Make predictions on the testing set
y_pred = best_model4.predict(X_test)

# Evaluate the model
accuracy_nbCV2 = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy_nbCV2)
print("Best Hyperparameters:", best_params4)
print(f"Classification Report:\n{classification_report}")


# In[49]:


import seaborn as sns
import matplotlib.pyplot as plt

models = ['Logistic Regression','Naive Bayes', 'Random Forest','Logistic Regression after CV','Naive Bayes after CV']
accuracies = [accuracy_lr2, accuracy_nb2, accuracy_rf2,accuracy_lrCV2, accuracy_nbCV2]

# Set a color palette for the models
colors = sns.color_palette('Set2', n_colors=len(models))

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=accuracies, y=models, palette=colors)

ax.set_title('Model Accuracies BoW', fontsize=16)
ax.set_xlabel('Accuracy', fontsize=14)
ax.set_ylabel('Models', fontsize=14)
ax.set_xlim(0, 1)
ax.tick_params(axis='both', labelsize=12)

# Add the accuracy values on each bar
for i, acc in enumerate(accuracies):
    ax.text(acc+0.01, i, f'{acc:.4f}', va='center', fontsize=12)

plt.show()


# In[61]:


import numpy as np
import matplotlib.pyplot as plt

# Existing code
models = ['Logistic Regression', 'Naive Bayes', 'Random Forest', 'Logistic Regression after CV', 'Naive Bayes after CV']
tfidf_acc = [accuracy_lr, accuracy_nb, accuracy_rf, accuracy_lrCV, accuracy_nbCV]
bow_acc = [accuracy_lr2, accuracy_nb2, accuracy_rf2, accuracy_lrCV2, accuracy_nbCV2]
x = np.arange(len(models))

fig, ax = plt.subplots(figsize=(15,6))
ax.bar(x - 0.2, tfidf_acc, width=0.4, label='TF-IDF', color='blue')
ax.bar(x + 0.2, bow_acc, width=0.4, label='Bag-of-Words', color='black',alpha=0.5)
ax.set_title('Model Accuracies Comparison', fontsize=16)
ax.set_xlabel('Models', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.tick_params(axis='both', labelsize=12)
plt.xticks(x, models, rotation=45, ha='right')

# Adding accuracy values to each bar
for i, v in enumerate(tfidf_acc):
    ax.text(i - 0.2, v + 0.01, str(round(v, 4)), color='black', ha='center', fontsize=12)

for i, v in enumerate(bow_acc):
    ax.text(i + 0.2, v + 0.01, str(round(v, 4)), color='black', ha='center', fontsize=12)

plt.show()


# In[43]:


## Evaluate the model's performance on unseen data

import pandas as pd
# Load the new unseen CSV file data
new_data = pd.read_csv(r"C:\Users\ashu1\OneDrive\Documents\new_sentiment.csv")


# In[44]:


# Apply preprocessing to the "text" column of new data
new_data['text'] = new_data['text'].apply(preprocess_text)


# In[45]:


# Transform the new data using the trained TF-IDF vectorizer
X_new = tfidf.transform(new_data['text'])


# In[46]:


# Make predictions on the new data using the best model
new_predictions = best_model1.predict(X_new)


# In[47]:


# Store the predictions in a new column of the new_data DataFrame
new_data['predicted_label'] = new_predictions


# In[52]:


new_data.head(10)


# In[ ]:




