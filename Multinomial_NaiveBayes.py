# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 11:44:42 2025

@author: avina
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Load the dataset
data = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Foundational ML Algorithms II/lNlhedMcSH63Idvr6lzE_valid.csv')  # Replace with the actual file path
data.head()
data.shape
data.columns

# Step 1: Data Preprocessing
# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Encode the labels
data['label'] = data['label'].astype('category')  
data['label_encoded'] = data['label'].cat.codes   

X_text = data['text']
y = data['label_encoded']

# Split the dataset
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.3, random_state=42)

# Define feature extraction techniques
vectorizers = {
    "CountVectorizer": CountVectorizer(stop_words='english', max_features=5000),
    "TF-IDF": TfidfVectorizer(stop_words='english', max_features=5000),
    "CountVectorizer with Bigrams": CountVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2)),
}

# Define Naive Bayes variants
nb_variants = {
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
}

# Iterate through vectorizers and Naive Bayes variants
results = []
for vec_name, vectorizer in vectorizers.items():
    # Apply feature extraction
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    for nb_name, nb_model in nb_variants.items():
        # Train and evaluate the model
        nb_model.fit(X_train, y_train)
        y_pred = nb_model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')  
        results.append({"Vectorizer": vec_name, "Model": nb_name, "Accuracy": acc, "F1 Score": f1})
        
        # Display detailed metrics
        print(f"\n=== {vec_name} with {nb_name} ===")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print(f"Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")

results_df = pd.DataFrame(results)

# Display results
print("\nModel Performance Summary:")
print(results_df)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.barplot(data=results_df, x="Vectorizer", y="F1 Score", hue="Model")
plt.title("Performance of Naive Bayes Variants with Different Vectorizers")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)
plt.legend(title="Model")
plt.show()

# Hyperparameter Tuning for BernoulliNB


# Use the best vectorizer for BernoulliNB 
best_vectorizer_name = "CountVectorizer with Bigrams"  
best_vectorizer = vectorizers[best_vectorizer_name]
X_train = best_vectorizer.fit_transform(X_train_text)
X_test = best_vectorizer.transform(X_test_text)

# Define parameter grid
param_grid = {
    'alpha': [0.1, 0.5, 1, 5, 10],  # Smoothing parameter
    'binarize': [None, 0.0, 0.5, 1.0],  # Threshold for binarization
}

# Perform GridSearchCV
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(BernoulliNB(), param_grid, scoring='f1_weighted', cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and performance
print("\nBest Parameters for BernoulliNB:", grid_search.best_params_)
best_bernoulli_model = grid_search.best_estimator_

# Evaluate the best BernoulliNB model
y_pred_tuned = best_bernoulli_model.predict(X_test)
print("\nConfusion Matrix (Tuned BernoulliNB):\n", confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report (Tuned BernoulliNB):\n", classification_report(y_test, y_pred_tuned))
print(f"Tuned BernoulliNB F1 Score: {f1_score(y_test, y_pred_tuned, average='weighted'):.2f}")

# Define the parameter grid for MultinomialNB
param_grid2 = {
    'alpha': [0.1, 0.5, 1.0, 5.0, 10.0],  # Smoothing parameter
}

# Perform GridSearchCV
grid_search = GridSearchCV(MultinomialNB(), param_grid2, scoring='f1_weighted', cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Display the best parameters
print("\nBest Parameters for MultinomialNB:", grid_search.best_params_)

# Use the best model for predictions
best_multinomial_model = grid_search.best_estimator_
y_pred_tuned = best_multinomial_model.predict(X_test)

# Evaluate the best MultinomialNB model
print("\nConfusion Matrix (Tuned MultinomialNB):\n", confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report (Tuned MultinomialNB):\n", classification_report(y_test, y_pred_tuned))
print(f"Tuned MultinomialNB F1 Score: {f1_score(y_test, y_pred_tuned, average='weighted'):.2f}")
print(f"Tuned MultinomialNB Accuracy Score: {accuracy_score(y_test, y_pred_tuned):.2f}")