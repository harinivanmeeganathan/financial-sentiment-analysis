import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define a text-cleaning function
def clean_text(text):
    """
    Cleans input text by removing special characters, lowercasing, and removing stopwords.
    """
    # Lowercase text
    text = text.lower()

    # Remove special characters, numbers, and punctuations
    text = re.sub(r"[^a-zA-Z]+", " ", text)

    # Tokenize text
    words = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Rejoin words into a cleaned sentence
    return " ".join(words)

# Load the dataset, clean it, and save it
def load_and_clean_data(file_path):
    """
    Loads a CSV file, applies text cleaning, and saves the cleaned DataFrame.
    """
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the data file
    file_path = os.path.join(current_dir, "../data", file_path)

    # Load the dataset
    df = pd.read_csv(file_path, encoding="latin-1")

    # Display basic information about the dataset
    print("Original Data:")
    print(df.head())
    print(df.info())

    # Rename columns if necessary
    df.columns = ["sentiment", "text"]

    # Clean text column
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Save cleaned data to the data folder
    cleaned_file_path = os.path.join(current_dir, "../data/cleaned_data.csv")
    df.to_csv(cleaned_file_path, index=False)

    print(f" Cleaned data saved to {cleaned_file_path}")

    return df

if __name__ == "__main__":
    cleaned_data = load_and_clean_data("data.csv")
    print(cleaned_data.head())
