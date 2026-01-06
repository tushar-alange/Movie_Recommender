from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
movies = pd.read_csv("movies.csv")

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])
cosine_sim = cosine_similarity(tfidf_matrix)

# Recommendation function
def get_recommendations(title):
    title_lower = title.lower()
    movie_titles = movies['title'].str.lower()
    matches = movie_titles[movie_titles.str.contains(title_lower)]
    if matches.empty:
        return [{"title": "Movie not found", "genres": ""}]
    
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    
    recommendations = []
    for i in [i[0] for i in sim_scores]:
        recommendations.append({
            "title": movies['title'].iloc[i],
            "genres": movies['tags'].iloc[i]  # or 'genres' column if available
        })
    return recommendations


# Home page
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        movie_name = request.form["movie"]
        recommendations = get_recommendations(movie_name)
        return render_template("result.html", recommendations=recommendations)
    return render_template("index.html")

# API route to get all movie titles
@app.route("/movies_list")
def movies_list():
    titles = movies['title'].tolist()
    return jsonify(titles)

if __name__ == "__main__":
    app.run(debug=True)
