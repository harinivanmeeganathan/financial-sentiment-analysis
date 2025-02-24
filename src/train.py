import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Constructed the absolute path dynamically
def get_cleaned_data_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "../data/cleaned_data.csv")

# Load and preprocess the dataset
def load_data():
    file_path = get_cleaned_data_path()
    df = pd.read_csv(file_path)
    
    # Debugging: Check column names
    print("Columns in DataFrame:", df.columns)
    
    df = df[['cleaned_text', 'sentiment']]
    label_map = {"positive": 1, "negative": 0, "neutral": 2}
    df['labels'] = df['sentiment'].map(label_map)
    
    return df

# Train the Hugging Face model
def train_huggingface_model(data):
    """
    Trains a Hugging Face transformer model (e.g., distilBERT) for sentiment analysis.
    """
    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(data)

    # Ensure `cleaned_text` is formatted correctly for batch processing
    dataset = dataset.map(lambda x: {"cleaned_text": [str(t) for t in x["cleaned_text"]]}, batched=True)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True)

    # Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Split into train and test datasets
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="../models/",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="../logs/",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda pred: {"accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1))},
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("../models/sentiment_model")
    tokenizer.save_pretrained("../models/sentiment_model")

    # Evaluate the model
    predictions = trainer.predict(test_dataset)
    predicted_classes = predictions.predictions.argmax(-1)
    print("Accuracy:", accuracy_score(test_dataset["labels"], predicted_classes))
    print("Classification Report:")
    print(classification_report(test_dataset["labels"], predicted_classes))
    print("Confusion Matrix:")
    print(confusion_matrix(test_dataset["labels"], predicted_classes))

    return model, tokenizer

if __name__ == "__main__":
    # Load cleaned data
    data = load_data()

    # Train the Hugging Face model
    model, tokenizer = train_huggingface_model(data)