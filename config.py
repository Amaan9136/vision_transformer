"""
Configuration for the Vision Transformer application.
"""

import os
from flask import Flask

# Initialize Flask app
flask_app = Flask(__name__)
flask_app.config['UPLOAD_FOLDER'] = 'static/uploads'
flask_app.config['ATTENTION_MAPS_FOLDER'] = 'static/attention_maps'
flask_app.config['FEATURE_MAPS_FOLDER'] = 'static/feature_maps'
flask_app.config['TRANSFORMATION_FOLDER'] = 'static/transformations'
flask_app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directories exist
os.makedirs(flask_app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(flask_app.config['ATTENTION_MAPS_FOLDER'], exist_ok=True)
os.makedirs(flask_app.config['FEATURE_MAPS_FOLDER'], exist_ok=True)
os.makedirs(flask_app.config['TRANSFORMATION_FOLDER'], exist_ok=True)