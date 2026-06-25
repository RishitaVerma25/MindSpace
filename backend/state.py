"""
state.py — Shared in-memory application state.

All route Blueprints import from here so that a single copy of each variable
is maintained across the entire app.
"""

import sys
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class AppState:
    def __init__(self):
        self.sia = None
        self.data_df = None
        self.corr_matrix = None
        self.compare_df = None
        self.compare_meta = None
        self.eval_metrics = {'primary': None, 'compare': None}

    def get_sia(self):
        if self.sia is None:
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        return self.sia

sys.modules[__name__] = AppState()
