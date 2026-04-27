import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Download stopwords if not already installed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Cleans the raw text data."""
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', str(text))
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def main():
    print("Loading dataset...")
    # Assuming your dataset has 'text' for the article and 'label' for REAL/FAKE
    try:
        df = pd.read_csv('news.csv')
    except FileNotFoundError:
        print("Error: Please download the dataset and save it as 'news.csv' in this directory.")
        return

    # Drop any rows with missing text
    df = df.dropna(subset=['text', 'label'])

    print("Cleaning text data... (This might take a minute)")
    df['clean_text'] = df['text'].apply(clean_text)

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

    print("Extracting features using TF-IDF...")
    # Initialize a TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7)

    # Fit and transform the training data, transform the testing data
    tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
    tfidf_test = tfidf_vectorizer.transform(X_test)

    print("Training the PassiveAggressiveClassifier...")
    # Initialize the classifier
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    print("Evaluating the model...")
    # Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    
    print(f'\n--- Results ---')
    print(f'Accuracy: {round(score*100, 2)}%')
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

if __name__ == '__main__':
    main()
