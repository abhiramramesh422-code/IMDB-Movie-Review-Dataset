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

The results_df provides a concise summary of the performance of the four models trained on the IMDb movie review dataset: Logistic Regression, SVM, Naive Bayes, and Random Forest. The metrics included are Train Accuracy, Test Accuracy, Precision, Recall, and F1-Score.

Logistic Regression: This model shows a good balance between training and test accuracy (0.9105 vs 0.8898). It also achieves the highest scores across all evaluation metrics on the test set: Test Accuracy (0.8898), Precision (0.8837), Recall (0.8978), and F1-Score (0.8907). This indicates that Logistic Regression is the best-performing model among the ones tested for this task.

SVM (LinearSVC): The SVM model has a slightly higher training accuracy (0.9273) than Logistic Regression, but its test accuracy (0.8812) is slightly lower. Its precision, recall, and F1-score on the test set are also slightly lower than Logistic Regression (0.8761, 0.8880, and 0.8820 respectively). This suggests that while SVM fits the training data well, it might not generalize quite as effectively to unseen data as Logistic Regression.

Naive Bayes (MultinomialNB): The Naive Bayes model has the lowest training accuracy (0.8612) and test accuracy (0.8531) among the four models. Its precision,
recall, and F1-score on the test set (0.8480, 0.8604, and 0.8542 respectively) are also the lowest. This indicates that for this dataset and feature representation, Naive Bayes is the least effective model.

Random Forest: The Random Forest model exhibits perfect training accuracy (1.0000), which is a strong indicator of overfitting. Despite this, its test accuracy (0.8568) is slightly better than Naive Bayes but significantly lower than Logistic Regression and SVM. Its precision (0.8654) is relatively good, but its recall (0.8450) and F1-score (0.8551) are lower than Logistic Regression and SVM. The high training accuracy and lower test performance suggest that the Random Forest model is not generalizing well to unseen data due to overfitting.

In conclusion, based on the results_df, the Logistic Regression model demonstrates the best overall performance on the test set, achieving the highest scores across all the evaluated metrics. The SVM model is the second-best performer, while Naive Bayes and Random Forest perform considerably worse, with Random Forest showing clear signs of overfitting.

HOW TO USE:

Ensure dependencies for Python libraries like scikit-learn, pandas, matplotlib, and seaborn are installed.

Run the notebook to load and preprocess the IMDb dataset.

Train the models or use saved models to predict sentiments on new text.













