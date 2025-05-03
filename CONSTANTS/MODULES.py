# -------------------------
# Built-in modules
# -------------------------
import os
import re
import time
import uuid
import json
import base64
import random
import logging
import http.client
import urllib.parse
from io import BytesIO
from datetime import datetime
import concurrent.futures
import string, socketio

# -------------------------
# Third-party modules
# -------------------------
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import torch
import imagehash
import cv2
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import chromadb

# -------------------------
# Flask & SocketIO
# -------------------------
from flask import Flask, render_template, request, jsonify, url_for, send_file
from flask_socketio import SocketIO, emit

# -------------------------
# Hugging Face Transformers
# -------------------------
from transformers import (
    ViTForImageClassification, 
    ViTImageProcessor,
    ViTModel,
    AutoFeatureExtractor
)

# -------------------------
# LangChain
# -------------------------
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# -------------------------
# Logging configuration
# -------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
