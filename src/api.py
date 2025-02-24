from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
import os
import jwt
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv


# Initialize FastAPI app
app = FastAPI()

# Load model and tokenizer
MODEL_PATH = "models/sentiment_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Set model to evaluation mode
model.eval()

# Define labels
LABELS = {0: "Negative", 1: "Positive", 2: "Neutral"}

# Load environment variables from .env
load_dotenv()

# Fetch secret key and other configurations from .env
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Function to create access token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Authentication endpoint (mock user login)
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Mock authentication (Replace with real authentication logic)
    if form_data.username != "admin" or form_data.password != "password":
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Token verification function
def verify_token(token: str = Security(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Prediction function
def predict_sentiment(text: str):
    """Predict sentiment of a given text."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return LABELS[prediction]

# API endpoint for sentiment analysis with authentication
from pydantic import BaseModel

# Define a request body model
class SentimentRequest(BaseModel):
    text: str

# API endpoint
@app.post("/predict/")
async def analyze_sentiment(request: SentimentRequest, token: str = Depends(verify_token)):
    text = request.text
    sentiment = predict_sentiment(text)
    return {"text": text, "sentiment": sentiment}


# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
