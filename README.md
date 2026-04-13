# Stock Sentiment Tracker

A full-stack financial news sentiment analysis tool that compares VADER and FinBERT on real financial headlines.

---

## Overview

General-purpose NLP models struggle with financial language. This project quantifies that gap by evaluating VADER against FinBERT on 30 manually labeled financial headlines.

| Model | Accuracy |
|---|---|
| VADER | 56.7% (17/30) |
| FinBERT | 76.7% (23/30) |

---

## Tech Stack

- **Backend** — Python, Flask, VADER (NLTK), FinBERT (HuggingFace Transformers), pandas
- **Frontend** — React, Axios

---

## Setup

**Backend**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Frontend**
```bash
cd frontend
npm install
npm start
```

**Run model evaluation**
```bash
cd backend
python evaluate_labeled_headlines.py
```

---

## Key Findings

- FinBERT outperforms VADER by 20 percentage points on financial text
- Neutral headlines are the hardest category for both models
- Domain-specific models significantly outperform general NLP tools in finance

---
