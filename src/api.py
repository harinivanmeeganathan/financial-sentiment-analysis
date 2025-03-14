from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
import os
import jwt
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
import logging
from sqlalchemy.orm import Session
import sys
from pathlib import Path
from pydantic import BaseModel

# Append project path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import SessionLocal
from src.utils import verify_password, get_user_by_username

# Load environment variables
load_dotenv()

# Hugging Face Authentication Token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Secret Key for JWT
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# Set up logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "api_logs.log")
logging.basicConfig(
    filename=LOG_FILE, 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI app
app = FastAPI()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Load model lazily
MODEL_NAME = "harinivanmeeganathan/financial-sentiment-model"
model = None
tokenizer = None

def get_model():
    """Load the model and tokenizer lazily to save memory."""
    global model, tokenizer
    if model is None or tokenizer is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, device_map="auto", load_in_8bit=True, use_auth_token=HF_TOKEN
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
    return model, tokenizer

# Define labels
LABELS = {0: "Negative", 1: "Positive", 2: "Neutral"}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Token creation function
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Token verification function
def verify_token(token: str = Security(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Admin authentication check
def admin_required(token_data: dict = Depends(verify_token)):
    if token_data.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")

# Authentication endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user_by_username(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}

# Protected route: Admin-only
@app.get("/admin-only/")
async def admin_route(token_data: dict = Depends(admin_required)):
    return {"message": "Welcome, Admin! You have access to this route."}

# Protected route: User-only
@app.get("/user-only/")
async def user_route(token_data: dict = Depends(verify_token)):
    return {"message": f"Welcome, {token_data.get('sub')}! You have user access."}

# Define request body model for sentiment prediction
class SentimentRequest(BaseModel):
    text: str

# Prediction function
def predict_sentiment(text: str):
    """Predict sentiment of a given text."""
    model, tokenizer = get_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return LABELS[prediction]

# API endpoint for sentiment analysis with authentication and rate limiting
@app.post("/predict/")
@limiter.limit("100/minute")  # Limit to 100 requests per minute
async def analyze_sentiment(request: Request, input_data: SentimentRequest, token: str = Depends(verify_token)):
    text = input_data.text
    sentiment = predict_sentiment(text)
    
    # Log prediction
    logging.info(f"User: {token['sub']} | Input Text: {text} | Sentiment: {sentiment}")
    
    return {"text": text, "sentiment": sentiment}

# Exception handler for rate limit
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please try again later."}
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unexpected Error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "An internal server error occurred. Please try again later."}
    )

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
