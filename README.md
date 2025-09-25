# IMDB-Movie-Review-Dataset

Overview

This project applies machine learning techniques to the IMDb movie review dataset in order to classify reviews as either positive or negative. The dataset consists of textual movie reviews with corresponding sentiment labels. The goal is to compare the performance of multiple models using standard evaluation metrics.

Dataset

File: IMDB Dataset.csv

Columns:

review: The text content of the movie review

sentiment: The label associated with the review (positive or negative)

The dataset contains an equal distribution of positive and negative reviews, ensuring balance for binary classification.

Methodology

Preprocessing

Sentiment labels were converted to numerical values (positive → 1, negative → 0).

Text data was transformed into numerical features using TF-IDF Vectorization with a vocabulary size limited to the top 5000 terms.

Models Evaluated

Logistic Regression

Linear Support Vector Machine (SVM)

Multinomial Naive Bayes

Random Forest Classifier

Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Key Observations

Linear SVM consistently achieved the highest performance across all metrics.

Logistic Regression provided results comparable to SVM and served as a strong baseline.

Naive Bayes was computationally efficient but showed lower accuracy compared to SVM and Logistic Regression.

Random Forest performed moderately well but was slower due to the ensemble method.

Visualizations

Confusion Matrices: To illustrate true vs. predicted classifications for each model.

Comparison Bar Chart: To compare Accuracy, Precision, Recall, and F1-score across models in a single plot.

How to Run

Upload IMDB Dataset.csv to the working environment (Google Colab or Jupyter Notebook).

Open one of the provided notebooks:

imdb_sentiment_ready.ipynb (evaluates models with accuracy and confusion matrices)

imdb_sentiment_metrics.ipynb (evaluates models using accuracy, precision, recall, and F1-score)

Execute all cells to reproduce the results.

Conclusion

This project demonstrates the application of classical machine learning methods to sentiment classification tasks. Linear SVM emerged as the best-performing model for this dataset, followed closely by Logistic Regression. The analysis highlights the trade-offs between accuracy and computational efficiency across different algorithms.












