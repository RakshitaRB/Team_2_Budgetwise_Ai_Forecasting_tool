# utils/auth.py
import json
import os
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from jose import JWTError, jwt

# Secret key for JWT tokens
SECRET_KEY = 'your-secret-key-here-change-in-production-12345'
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# A fixed salt for simplicity (in production, use per-user salts)
PEPPER = "budget-forecasting-pepper-2024"

def verify_password(plain_password, hashed_password):
    """Verify a password against its hash"""
    try:
        return hmac.compare_digest(
            hashed_password,
            get_password_hash(plain_password)
        )
    except:
        return False

def get_password_hash(password):
    """Create a password hash using multiple rounds of hashing"""
    if not password:
        return ""
    
    # Convert to string and limit length
    password_str = str(password)[:100] + PEPPER
    
    # Multiple rounds of hashing for basic security
    hash_result = password_str
    for _ in range(1000):  # 1000 rounds
        hash_result = hashlib.sha256(hash_result.encode()).hexdigest()
    
    return hash_result

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None