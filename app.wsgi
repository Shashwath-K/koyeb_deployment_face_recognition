# app.wsgi
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app as application

if __name__ == "__main__":
    application.run()