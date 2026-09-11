# EE8223 Deep Learning — Assignments

Coursework from EE8223 (Deep Learning), covering convolution fundamentals, recurrent networks, CNN image classification, and Transformer self-attention — implemented from scratch and with TensorFlow/Keras.

## Contents

### Assignment 1 (`Assignment-1/Assignment_1.ipynb`)
2D convolution implemented from scratch with NumPy:
- Single filter / single feature map with configurable stride.
- Multiple filters producing a stack of feature maps.
- Multi-channel input convolved with multiple filters (a minimal from-scratch conv layer, stage-by-stage).

### Assignment 2 (`Assignment-2/`)
- **`assignment_2_deep_learning.ipynb`** — a mixed problem set:
  - RNN model design in Keras (`SimpleRNN` + `TimeDistributed(Dense)`) for multi-step, multi-feature time-series output.
  - Two short Python/NumPy indexing exercises.
  - A two-class CNN image classifier (baby vs. dog), trained both with and without pooling/subsampling, comparing validation accuracy between the two architectures.
  - Next-day stock price prediction for Toyota (TM) using 5 correlated auto-sector tickers (TM, F, GM, HMC, TSLA) as inputs, comparing a `SimpleRNN` against an `LSTM`.
  - Note: cell outputs containing embedded plot images were stripped to keep this file a reasonable size for the repo; all code, markdown, and text/log outputs (training curves as text, accuracy figures, etc.) are preserved.
- **`RandomToyotaMultipleTimeSeries.ipynb`** — a walkthrough notebook building a `SimpleRNN` that predicts one time series from several correlated time series (the windowing/reshaping approach referenced in the assignment above).

### Assignment 3 (`Assignment-3/Assignment_3_Samiul_Islam.ipynb`)
Transformer self-attention implemented from scratch with NumPy:
- Word + sinusoidal positional embeddings.
- Query/Key/Value projection matrices.
- Scaled dot-product attention (`softmax(QKᵀ / √d) · V`) computed step-by-step with shape annotations at each stage.

## Tech stack
- Python, NumPy
- TensorFlow / Keras
- yfinance, pandas, scikit-learn
- Jupyter / Google Colab
