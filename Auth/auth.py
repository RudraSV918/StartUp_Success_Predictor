import pandas as pd
import hashlib
import os

FILE_PATH = "Data/users.csv"

def load_users():
    if os.path.exists(FILE_PATH):
        return pd.read_csv(FILE_PATH)
    else:
        return pd.DataFrame(columns=["username","password"])
    
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup_user(username,password):
    users = load_users()

    if username in users["username"].values:
        return False
    
    new_user = pd.DataFrame([{
        "username": username,
        "password": hash_password(password)
    }])

    users = pd.concat([users,new_user], ignore_index=True)
    users.to_csv(FILE_PATH, index=False)
    return True

def login_user(username,password):
    users = load_users()
    hashed = hash_password(password)

    user = users[
        (users["username"] == username) &
        (users["password"] == hashed)
    ]
    return not user.empty
