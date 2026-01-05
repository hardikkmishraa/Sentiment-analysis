# Sentiment Analysis on IMDB Reviews with LSTM

This project implements a binary sentiment analysis model using Long Short-Term Memory (LSTM) networks on the IMDB dataset of 50K movie reviews.

The model achieves a test accuracy of approximately 88.65% and test loss of 0.303 after training for 3 epochs.

## Features

- Downloads and processes the IMDB dataset from Kaggle (50,000 reviews, balanced positive/negative).
- Text tokenization with top 5,000 words and padding to length 200.
- LSTM model with embedding (128 dims), LSTM (128 units, dropout 0.2), and sigmoid output.
- Includes a predict function for new reviews, classifying as Positive or Negative based on 0.5 threshold.

## Usage

Run the notebook in Google Colab with GPU enabled (T4 used in the file).

Example predictions:

- "This movie was fantastic. I loved it." → Positive
- "This movie was not that good." → Negative

## Requirements

- Python 3, TensorFlow/Keras, Pandas, Scikit-learn, Kaggle API (credentials via JSON).
- No additional installs needed beyond **`pip install kaggle`** if missing.

## Model Architecture

| **Layer** | **Output Shape** | **Parameters** |
| --- | --- | --- |
| Embedding | (None, 200, 128) | 640,000 |
| LSTM | (None, 128) | ~650,000 |
| Dense | (None, 1) | 129 |

Total params: ~853,400 (trainable).

## Training Results

- Epoch 1: Train acc 0.8863, Val acc 0.8665
- Epoch 2: Train acc 0.8959, Val acc 0.8504
- Epoch 3: Train acc 0.9171, Val acc 0.8766

