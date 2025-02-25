from passlib.context import CryptContext
from sqlalchemy.orm import Session
from database import User

# Hashing context for passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Function to hash passwords
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# Function to verify password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Function to get a user by username
def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

# Function to create a new user
def create_user(db: Session, username: str, password: str, role: str = "user"):
    hashed_password = hash_password(password)
    new_user = User(username=username, hashed_password=hashed_password, role=role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user
