import pandas as pd
import ast

# Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge on title
movies = movies.merge(credits, on="title")

# Functions to extract names from JSON
def extract_names(obj):
    try:
        data = ast.literal_eval(obj)
        return " ".join([item['name'] for item in data])
    except:
        return ""

# Clean columns
movies["genres"] = movies["genres"].apply(extract_names)
movies["cast"] = movies["cast"].apply(extract_names)
movies["keywords"] = movies["keywords"].apply(extract_names)
movies["overview"] = movies["overview"].fillna("")

# Create a final combined text column
movies["tags"] = (
    movies["overview"] + " " +
    movies["genres"] + " " +
    movies["cast"] + " " +
    movies["keywords"]
)

# Only keep title + tags
final = movies[["title", "tags"]]

# Save as movies.csv
final.to_csv("movies.csv", index=False)

print("movies.csv created successfully!")
