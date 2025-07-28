#!/usr/bin/env python3
"""
Download models script for Docker build
"""
import os
import shutil

# Set environment variables
os.environ['PYTHONPATH'] = '/app:/app/src'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('Downloading Round 1B models for offline execution...')

print('1. Downloading sentence-transformers model...')
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/models')
    print('   ✅ all-MiniLM-L6-v2 model downloaded')
except Exception as e:
    print(f'   ❌ Error downloading sentence-transformers model: {e}')
    raise

print('2. Downloading summarizer model...')
try:
    from summarizer import Summarizer
    summarizer = Summarizer('distilbert-base-uncased')
    print('   ✅ distilbert-base-uncased model downloaded')
except Exception as e:
    print(f'   ❌ Error downloading summarizer model: {e}')
    raise

print('🎉 All Round 1B models downloaded successfully!') 