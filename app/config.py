"""
Configuration for data sources and URLs.
Centralizes all data repository references.
"""

import os

# GitHub data repository information
GITHUB_USERNAME = os.getenv('GITHUB_USERNAME', 'kananashish')
DATA_REPO_NAME = os.getenv('DATA_REPO_NAME', 'heart-data')

# Construct GitHub raw content URL
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{DATA_REPO_NAME}/main"

# Data source URLs
DATA_URLS = {
    'demo_data': f"{GITHUB_RAW_BASE}/train_balanced.csv",
    'test_data': f"{GITHUB_RAW_BASE}/test.csv",
    'model_results': f"{GITHUB_RAW_BASE}/model_results.csv",
}

# Optional: For future expansion
OPTIONAL_DATA_URLS = {
    'combined_heart': f"{GITHUB_RAW_BASE}/combined_heart.csv",
    'original_heart': f"{GITHUB_RAW_BASE}/heart.csv",
    'original_train': f"{GITHUB_RAW_BASE}/train_original.csv",
}

# Timeout for GitHub requests (seconds)
GITHUB_TIMEOUT = 10

# Enable/disable fallback to hardcoded data
USE_FALLBACK_DATA = True
