# Fake-News-Detection
This project is a Natural Language Processing (NLP) model designed to classify news articles as either **Real** or **Fake**. It was built as part of an internship project for Codec Technologies.
## 🚀 Overview
The model uses a `TfidfVectorizer` to extract features from raw text data and a `PassiveAggressiveClassifier` to predict the legitimacy of the news article. 

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas
* **Natural Language Processing:** NLTK
* **Machine Learning:** Scikit-Learn

## ⚙️ How to Run
1. Clone this repository to your local machine.
2. Download the Fake News dataset from Kaggle and place it in the root directory as `news.csv`.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
