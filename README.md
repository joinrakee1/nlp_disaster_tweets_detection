# NLP Disaster Tweets Classification

This project uses Recurrent Neural Networks (RNNs) to classify tweets as either related to real disasters or not. The task involves building models that can understand short, informal text and make binary predictions based on context.

This project was completed as part of a **peer-graded assignment** for the **Introduction to Deep Learning** course and is submitted to the [Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started).

## Files

**Included in this repository:**
- `nlp_disaster_tweets_kaggle_mini_project`: Jupyter notebook containing EDA, text preprocessing, RNN model building, training, evaluation, and submission generation for Kaggle.
- `train.csv`: Training dataset with labeled tweets.
- `test.csv`: Test dataset with unlabeled tweets (for submission).
- `submission_baseline.csv`: Predictions from the baseline RNN model.
- `submission_bidirectional.csv`: Predictions from the bidirectional LSTM model (best performance).
- `submission_stacked.csv`: Predictions from the stacked LSTM model.

> All files are included for reproducibility and ease of review. You can run the notebook end-to-end without downloading additional files from Kaggle.

## Models
Three RNN models were trained and compared:
1. **Baseline RNN**: Embedding + Single LSTM layer, dropout = 0.5, learning rate = 0.001
2. **Variant 1**: Bidirectional LSTM with dropout = 0.3
3. **Variant 2**: Stacked LSTM with batch normalization, dropout = 0.5, learning rate = 0.0005

## Results

The **Bidirectional LSTM** model achieved the highest validation accuracy (~76.4%) in the final run. Validation curves and training logs for all three models are included in the notebook for comparison.

It is worth noting that in some previous runs, the **Baseline RNN** model achieved higher validation accuracy than the bidirectional and stacked variants. This variation is expected in deep learning workflows due to factors like random weight initialization, dropout behavior, and the absence of a fixed random seed. These small changes can influence the training dynamics, especially on smaller datasets. 


## Requirements
- Python 3.10
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Install dependencies with:
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
