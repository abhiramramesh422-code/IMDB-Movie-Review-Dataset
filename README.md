# IMDB-Movie-Review-Dataset

OVERVIEW:

This project is a sentiment analysis classifier built to automatically determine whether movie reviews from the IMDb dataset are positive or negative. Using several machine learning models, the goal is to predict the sentiment behind a given review text.

DATASET:

The dataset contains 50,000 movie reviews balanced evenly between positive and negative sentiments. It was sourced from IMDb and contains raw review text that needed preprocessing before training.

DATA PREPROCESSING:

The text was cleaned and transformed using TF-IDF vectorization to convert words and phrases into numerical features the models can understand.

The sentiment labels were converted into binary format (positive=1, negative=0).

The dataset was split into 80% training and 20% testing sets for evaluation.

MODELS TRAINED:

Four models were trained and compared on this task:

Logistic Regression

Support Vector Machine (SVM)

Naive Bayes

Random Forest

Each model was trained using the TF-IDF features and the corresponding sentiment labels.

EVALUATION METRICS:

Model performance was assessed using:

Accuracy

Precision

Recall

F1-Score

ROC curves were also plotted for comparison.

RESULTS:

Logistic Regression performed best on the IMDb dataset with the highest test accuracy (0.8898), precision, recall, and F1-score, showing strong generalization. SVM had slightly higher training accuracy but lower test performance, indicating less effective generalization. Naive Bayes had the lowest accuracies and metrics, making it the least effective model. Random Forest overfit perfectly on training data (accuracy 1.0) but had lower test accuracy (0.8568) and metrics, showing poor generalization due to overfitting. Overall, Logistic Regression is the best model, followed by SVM, while Naive Bayes and Random Forest performed worse.

HOW TO USE:

Ensure dependencies for Python libraries like scikit-learn, pandas, matplotlib, and seaborn are installed.
Run the notebook to load and preprocess the IMDb dataset.
Train the models or use saved models to predict sentiments on new text.













