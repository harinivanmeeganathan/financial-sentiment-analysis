from database import SessionLocal
from utils import create_user

# Create a new session
db = SessionLocal()

# Add users
create_user(db, "admin", "adminpassword", role="admin")
create_user(db, "user1", "userpassword", role="user")

# Close session
db.close()
