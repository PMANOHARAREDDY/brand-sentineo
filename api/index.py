import sys
import os

# Add the project root to sys.path so Flask can find app.py, templates/, static/
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from app import app
