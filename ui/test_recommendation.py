import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui.recommendation_engine import *

engine = RecommendationEngine()

user_id = 100

print("\nWatched Movies:\n")

for movie in engine.get_watched_movies(user_id):
    print(movie)

print("\nRecommendations:\n")

for movie in engine.recommend(user_id):
    print(movie)