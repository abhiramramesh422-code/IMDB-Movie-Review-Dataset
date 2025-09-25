# IMDb Movie Review Sentiment Analysis

This project applies machine learning techniques to the IMDb movie review dataset to classify reviews as positive or negative.

## Dataset

- **File:** `IMDB Dataset.csv`  
- **Columns:**  
  - `review`: Text content of the movie review  
  - `sentiment`: Label (positive or negative)  
- Balanced dataset with an equal number of positive and negative reviews.

## Methodology

**Preprocessing:**  
- Sentiment labels converted to numerical values (positive → 1, negative → 0)  
- Text transformed using TF-IDF Vectorization (top 5000 terms)

**Models Evaluated:**  
- Logistic Regression  
- Linear SVM  
- Multinomial Naive Bayes  
- Random Forest Classifier  

**Evaluation Metrics:**  
- Accuracy  
- Precision  
- Recall  
- F1-score  

## Results

| Model                  | Accuracy | Precision | Recall | F1-score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | ~0.88    | ~0.88     | ~0.88  | ~0.88    |
| Linear SVM             | ~0.89    | ~0.89     | ~0.89  | ~0.89    |
| Naive Bayes            | ~0.84    | ~0.84     | ~0.83  | ~0.83    |
| Random Forest          | ~0.85    | ~0.85     | ~0.85  | ~0.85    |

**Key Observations:**  
- Linear SVM achieved the highest performance.  
- Logistic Regression was comparable and a strong baseline.  
- Naive Bayes was fast but slightly less accurate.  
- Random Forest performed moderately but slower due to ensemble computation.

## Visualizations

- Confusion matrices for each model  
- Comparison bar chart for Accuracy, Precision, Recall, and F1-score  

## How to Run

1. Upload `IMDB Dataset.csv` to your working environment (Google Colab or Jupyter Notebook).  
2. Open one of the provided notebooks:  
   - `imdb_sentiment_ready.ipynb` – evaluates models with accuracy and confusion matrices  
   - `imdb_sentiment_metrics.ipynb` – evaluates models using all metrics  
3. Execute all cells to reproduce the results.  

## Conclusion

Classical machine learning methods can effectively classify IMDb reviews. Linear SVM performed best, followed closely by Logistic Regression, highlighting trade-offs between accuracy and computational efficiency.













