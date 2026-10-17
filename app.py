import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import json
import zlib
import base64
import urllib.parse

# 1. Page Configuration
st.set_page_config(
    page_title="CineMind | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Complete MovieLens 100k Compressed Genre Map
GENRES_B64 = """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"""

@st.cache_data
def get_all_genres():
    raw_json = zlib.decompress(base64.b64decode(GENRES_B64.strip().encode('utf-8'))).decode('utf-8')
    gmap = json.loads(raw_json)
    return {int(k): v for k, v in gmap.items()}

# 3. Model & Artifact Loader
@st.cache_resource
def load_model_artifacts():
    similarity = joblib.load("item_similarity.pkl")
    train_matrix = joblib.load("train_matrix.pkl")
    global_mean = float(joblib.load("global_mean.pkl"))
    movies_df = pd.read_csv("movie_info.csv")
    movies_df["movie_id"] = movies_df["movie_id"].astype(int)
    return similarity, train_matrix, global_mean, movies_df

try:
    item_similarity_df, train_matrix, global_mean, movie_info = load_model_artifacts()
    genre_mapping = get_all_genres()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please ensure all .pkl and .csv files are present.")
    st.stop()

# 4. Enriched Movie Catalog & Metadata
def clean_movie_title(raw_title):
    clean = re.sub(r'\s*\(\d{4}\)\s*$', '', str(raw_title)).strip()
    if clean.endswith(', The'):
        clean = 'The ' + clean[:-5]
    elif clean.endswith(', A'):
        clean = 'A ' + clean[:-3]
    elif clean.endswith(', An'):
        clean = 'An ' + clean[:-4]
    return clean

def extract_movie_year(raw_title):
    m = re.search(r'\((\d{4})\)', str(raw_title))
    return m.group(1) if m else "1995"

CURATED_METADATA = {
    50: {
        "title": "Star Wars: Episode IV - A New Hope",
        "clean_title": "Star Wars",
        "year": "1977",
        "age_rating": "PG",
        "runtime": "2h 01m",
        "director": "George Lucas",
        "cast": "Mark Hamill, Harrison Ford, Carrie Fisher",
        "synopsis": "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to rescue Princess Leia from Darth Vader and save the galaxy from the Empire's death machine.",
        "backdrop": "https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=1600&auto=format&fit=crop"
    },
    100: {
        "title": "Fargo",
        "clean_title": "Fargo",
        "year": "1996",
        "age_rating": "R",
        "runtime": "1h 38m",
        "director": "Joel Coen, Ethan Coen",
        "cast": "Frances McDormand, William H. Macy, Steve Buscemi",
        "synopsis": "Jerry Lundegaard's inept crime scheme unravels into hilarious chaos due to his bungling henchmen and the persistent work of pregnant Police Chief Marge Gunderson.",
        "backdrop": "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?q=80&w=1600&auto=format&fit=crop"
    },
    127: {
        "title": "The Godfather",
        "clean_title": "The Godfather",
        "year": "1972",
        "age_rating": "R",
        "runtime": "2h 55m",
        "director": "Francis Ford Coppola",
        "cast": "Marlon Brando, Al Pacino, James Caan",
        "synopsis": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son, sparking a generational struggle.",
        "backdrop": "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=1600&auto=format&fit=crop"
    },
    56: {
        "title": "Pulp Fiction",
        "clean_title": "Pulp Fiction",
        "year": "1994",
        "age_rating": "R",
        "runtime": "2h 34m",
        "director": "Quentin Tarantino",
        "cast": "John Travolta, Uma Thurman, Samuel L. Jackson",
        "synopsis": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and dark redemption.",
        "backdrop": "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=1600&auto=format&fit=crop"
    },
    98: {
        "title": "The Silence of the Lambs",
        "clean_title": "The Silence of the Lambs",
        "year": "1991",
        "age_rating": "R",
        "runtime": "1h 58m",
        "director": "Jonathan Demme",
        "cast": "Jodie Foster, Anthony Hopkins, Scott Glenn",
        "synopsis": "A young F.B.I. cadet must confide in an incarcerated cannibalistic killer to catch another serial murderer who skins his victims.",
        "backdrop": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?q=80&w=1600&auto=format&fit=crop"
    },
    1: {
        "title": "Toy Story",
        "clean_title": "Toy Story",
        "year": "1995",
        "age_rating": "G",
        "runtime": "1h 21m",
        "director": "John Lasseter",
        "cast": "Tom Hanks, Tim Allen, Don Rickles",
        "synopsis": "A traditional pull-string cowboy doll is profoundly threatened when a futuristic space ranger action figure usurps his position as top toy.",
        "backdrop": "https://images.unsplash.com/photo-1534447677768-be436bb09401?q=80&w=1600&auto=format&fit=crop"
    },
    313: {
        "title": "Titanic",
        "clean_title": "Titanic",
        "year": "1997",
        "age_rating": "PG-13",
        "runtime": "3h 14m",
        "director": "James Cameron",
        "cast": "Leonardo DiCaprio, Kate Winslet, Billy Zane",
        "synopsis": "A seventeen-year-old aristocrat falls in love with a kind but penniless artist aboard the luxurious, ill-fated R.M.S. Titanic.",
        "backdrop": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?q=80&w=1600&auto=format&fit=crop"
    },
    64: {
        "title": "The Shawshank Redemption",
        "clean_title": "The Shawshank Redemption",
        "year": "1994",
        "age_rating": "R",
        "runtime": "2h 22m",
        "director": "Frank Darabont",
        "cast": "Tim Robbins, Morgan Freeman, Bob Gunton",
        "synopsis": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency and unbreakable spirit.",
        "backdrop": "https://images.unsplash.com/photo-1533929736458-ca588d08c8be?q=80&w=1600&auto=format&fit=crop"
    }
}

@st.cache_data
def build_enriched_catalog():
    vote_counts = train_matrix.count(axis=0)
    avg_ratings = train_matrix.mean(axis=0)
    
    catalog = {}
    for _, row in movie_info.drop_duplicates("movie_id").iterrows():
        mid = int(row["movie_id"])
        raw_title = row["title"]
        clean = clean_movie_title(raw_title)
        year = extract_movie_year(raw_title)
        genres = genre_mapping.get(mid, ["Drama"])
        votes = int(vote_counts.get(mid, 0))
        mean_rating = float(avg_ratings.get(mid, global_mean))
        
        curated = CURATED_METADATA.get(mid, {})
        
        if mid in train_matrix.columns:
            m_ratings = train_matrix[mid].dropna()
            pos = (m_ratings >= 4).sum()
            tot = len(m_ratings)
            approval = int((pos / tot) * 100) if tot > 0 else 88
        else:
            approval = 88
            
        if "age_rating" in curated:
            age_rating = curated["age_rating"]
        elif any(g in ["Horror", "Crime"] for g in genres):
            age_rating = "R"
        elif any(g in ["Animation", "Children's"] for g in genres):
            age_rating = "G"
        elif any(g in ["Action", "Sci-Fi", "Thriller"] for g in genres):
            age_rating = "14+"
        else:
            age_rating = "PG-13"

        director = curated.get("director", "Visionary Director")
        cast = curated.get("cast", "Ensemble Cast, Acclaimed Actors")
        runtime = curated.get("runtime", f"{1 + (mid % 2)}h {20 + (mid * 3 % 40)}m")
        
        synopsis = curated.get(
            "synopsis",
            f"An iconic {' and '.join(genres[:2])} cinematic classic released in {year}. "
            f"Featuring compelling performances and striking visual storytelling that has captivated "
            f"audiences and critics alike."
        )
        
        catalog[mid] = {
            "movie_id": mid,
            "raw_title": raw_title,
            "title": curated.get("title", clean),
            "clean_title": clean,
            "year": year,
            "genres": genres,
            "primary_genre": genres[0] if genres else "Drama",
            "votes": votes,
            "avg_rating": round(mean_rating, 2),
            "star_10": round(mean_rating * 2, 1),
            "approval": approval,
            "age_rating": age_rating,
            "director": director,
            "cast": cast,
            "runtime": runtime,
            "synopsis": synopsis,
            "backdrop": curated.get("backdrop", "https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=1600&auto=format&fit=crop")
        }
    return catalog

catalog = build_enriched_catalog()

# 5. Aesthetic SVG Poster Generator
GENRE_THEMES = {
    "Action": {"bg1": "#2a0808", "bg2": "#140406", "acc": "#ef4444", "sub": "#f87171", "icon": "action"},
    "Sci-Fi": {"bg1": "#071e3d", "bg2": "#050e1a", "acc": "#00f2fe", "sub": "#38bdf8", "icon": "scifi"},
    "Drama": {"bg1": "#210f36", "bg2": "#0f071a", "acc": "#c084fc", "sub": "#a855f7", "icon": "drama"},
    "Comedy": {"bg1": "#382404", "bg2": "#1a1002", "acc": "#fbbf24", "sub": "#f59e0b", "icon": "comedy"},
    "Romance": {"bg1": "#3b0826", "bg2": "#1a0410", "acc": "#f472b6", "sub": "#ec4899", "icon": "romance"},
    "Thriller": {"bg1": "#1a172e", "bg2": "#0d0b17", "acc": "#818cf8", "sub": "#6366f1", "icon": "thriller"},
    "Horror": {"bg1": "#360505", "bg2": "#170101", "acc": "#dc2626", "sub": "#991b1b", "icon": "horror"},
    "Animation": {"bg1": "#062b2b", "bg2": "#031414", "acc": "#2dd4bf", "sub": "#14b8a6", "icon": "animation"},
    "Adventure": {"bg1": "#0c2c15", "bg2": "#051409", "acc": "#4ade80", "sub": "#22c55e", "icon": "adventure"},
    "Crime": {"bg1": "#1e2430", "bg2": "#0d1017", "acc": "#94a3b8", "sub": "#64748b", "icon": "crime"},
    "Mystery": {"bg1": "#1e1438", "bg2": "#0d091a", "acc": "#a78bfa", "sub": "#7c3aed", "icon": "mystery"},
    "War": {"bg1": "#292314", "bg2": "#141109", "acc": "#d97706", "sub": "#b45309", "icon": "war"},
    "Western": {"bg1": "#3b1f0c", "bg2": "#1c0d04", "acc": "#f97316", "sub": "#c2410c", "icon": "western"},
}

def render_svg_icon(icon_type, color):
    if icon_type == "scifi":
        return f'''
        <circle cx="150" cy="145" r="48" fill="none" stroke="{color}" stroke-width="1.5" stroke-opacity="0.3"/>
        <ellipse cx="150" cy="145" rx="64" ry="20" fill="none" stroke="{color}" stroke-width="1.8" transform="rotate(-25 150 145)"/>
        <circle cx="150" cy="145" r="26" fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="2"/>
        <circle cx="160" cy="136" r="4.5" fill="{color}"/>
        '''
    elif icon_type == "action":
        return f'''
        <circle cx="150" cy="145" r="50" fill="none" stroke="{color}" stroke-width="1.5" stroke-opacity="0.3" stroke-dasharray="6,4"/>
        <circle cx="150" cy="145" r="32" fill="none" stroke="{color}" stroke-width="2"/>
        <path d="M150 118 L150 132 M150 158 L150 172 M123 145 L137 145 M163 145 L177 145" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>
        <polygon points="150,132 155,141 164,145 155,149 150,158 145,149 136,145 145,141" fill="{color}"/>
        '''
    elif icon_type == "comedy":
        return f'''
        <circle cx="150" cy="145" r="42" fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="2"/>
        <circle cx="136" cy="137" r="4.5" fill="{color}"/>
        <circle cx="164" cy="137" r="4.5" fill="{color}"/>
        <path d="M132 152 Q150 172 168 152" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round"/>
        '''
    elif icon_type == "romance":
        return f'''
        <path d="M150 170 C122 142 118 122 136 122 C146 122 150 132 150 132 C150 132 154 122 164 122 C182 122 178 142 150 170 Z" fill="{color}" fill-opacity="0.25" stroke="{color}" stroke-width="2.5"/>
        '''
    elif icon_type == "thriller":
        return f'''
        <path d="M116 145 C126 126 174 126 184 145 C174 164 126 164 116 145 Z" fill="none" stroke="{color}" stroke-width="2.5"/>
        <circle cx="150" cy="145" r="14" fill="{color}" fill-opacity="0.3" stroke="{color}" stroke-width="2"/>
        <circle cx="150" cy="145" r="4.5" fill="{color}"/>
        '''
    else:
        return f'''
        <circle cx="150" cy="145" r="42" fill="none" stroke="{color}" stroke-width="2"/>
        <circle cx="150" cy="145" r="11" fill="{color}" fill-opacity="0.3"/>
        <circle cx="150" cy="122" r="6" fill="{color}" fill-opacity="0.6"/>
        <circle cx="150" cy="168" r="6" fill="{color}" fill-opacity="0.6"/>
        <circle cx="127" cy="145" r="6" fill="{color}" fill-opacity="0.6"/>
        <circle cx="173" cy="145" r="6" fill="{color}" fill-opacity="0.6"/>
        '''

@st.cache_data
def get_svg_poster(mid):
    m = catalog.get(mid)
    if not m:
        return ""
    title = m["clean_title"]
    year = m["year"]
    genres = m["genres"]
    genre = genres[0] if genres else "Drama"
    theme = GENRE_THEMES.get(genre, {"bg1": "#171a26", "bg2": "#0c0e14", "acc": "#38bdf8", "sub": "#0284c7", "icon": "default"})
    
    words = title.split()
    lines = []
    curr = []
    for w in words:
        curr.append(w)
        if len(" ".join(curr)) > 14:
            if len(curr) > 1:
                curr.pop()
                lines.append(" ".join(curr))
                curr = [w]
            else:
                lines.append(" ".join(curr))
                curr = []
    if curr:
        lines.append(" ".join(curr))
    lines = lines[:3]
    
    title_svg = ""
    y_start = 245 if len(lines) == 1 else (232 if len(lines) == 2 else 218)
    for i, line in enumerate(lines):
        escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        title_svg += f'<text x="150" y="{y_start + i*26}" font-family="system-ui, -apple-system, sans-serif" font-weight="800" font-size="19" fill="#ffffff" text-anchor="middle" letter-spacing="0.4">{escaped}</text>'

    genre_str = " • ".join(genres[:2]) if genres else "CINEMA"
    escaped_genre = genre_str.replace('&', '&amp;').upper()
    icon_markup = render_svg_icon(theme["icon"], theme["acc"])
    
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 450" width="300" height="450">
  <defs>
    <radialGradient id="grad_{mid}" cx="50%" cy="28%" r="75%">
      <stop offset="0%" stop-color="{theme['bg1']}"/>
      <stop offset="55%" stop-color="{theme['bg2']}"/>
      <stop offset="100%" stop-color="#07080c"/>
    </radialGradient>
  </defs>
  <rect width="300" height="450" rx="14" fill="url(#grad_{mid})"/>
  <rect x="1" y="1" width="298" height="448" rx="13" fill="none" stroke="{theme['acc']}" stroke-width="1.2" stroke-opacity="0.35"/>
  <g transform="translate(18, 20)">
    <rect width="72" height="20" rx="10" fill="{theme['acc']}" fill-opacity="0.16" stroke="{theme['acc']}" stroke-opacity="0.4" stroke-width="1"/>
    <text x="36" y="14" font-family="system-ui, sans-serif" font-size="9" font-weight="700" fill="{theme['acc']}" text-anchor="middle" letter-spacing="1">{genre.upper()}</text>
  </g>
  <g transform="translate(225, 20)">
    <rect width="55" height="20" rx="10" fill="#ffffff" fill-opacity="0.08"/>
    <text x="27.5" y="14" font-family="system-ui, sans-serif" font-size="10" font-weight="600" fill="#94a3b8" text-anchor="middle">{year}</text>
  </g>
  {icon_markup}
  {title_svg}
  <text x="150" y="322" font-family="system-ui, sans-serif" font-size="11" font-weight="600" fill="{theme['acc']}" text-anchor="middle" letter-spacing="1.5">{escaped_genre}</text>
  <g transform="translate(100, 350)">
    <rect width="100" height="28" rx="14" fill="#000000" fill-opacity="0.55" stroke="#fbbf24" stroke-opacity="0.4" stroke-width="1"/>
    <path d="M18 7 L19.8 11.5 L24.5 11.8 L20.8 15 L21.9 19.6 L18 17 L14.1 19.6 L15.2 15 L11.5 11.8 L16.2 11.5 Z" fill="#fbbf24"/>
    <text x="60" y="18" font-family="system-ui, sans-serif" font-size="13" font-weight="700" fill="#f8fafc" text-anchor="middle">{m['avg_rating']:.1f}/5.0</text>
  </g>
  <line x1="40" y1="410" x2="260" y2="410" stroke="#ffffff" stroke-opacity="0.08" stroke-width="1"/>
  <text x="150" y="428" font-family="system-ui, sans-serif" font-size="10" font-weight="800" fill="#475569" text-anchor="middle" letter-spacing="3">CINEMIND</text>
</svg>'''
    return f"data:image/svg+xml;utf8,{urllib.parse.quote(svg)}"

# 6. Recommendation Engine (Vectorized & Hybrid)
def find_similar_movies(movie_id, n=6):
    if movie_id not in item_similarity_df.index:
        return []
    scores = item_similarity_df[movie_id].drop(movie_id, errors='ignore').sort_values(ascending=False).head(n)
    results = []
    for mid, sim in scores.items():
        if mid in catalog:
            results.append({
                "movie_id": mid,
                "score": float(sim),
                "match_pct": min(99, max(65, int(sim * 100)))
            })
    return results

def predict_rating_fast(user_id, movie_ids):
    if user_id not in train_matrix.index:
        return np.full(len(movie_ids), global_mean)
    user_ratings = train_matrix.loc[user_id].dropna()
    watched = user_ratings.index
    sub_sim = item_similarity_df.loc[movie_ids, watched].clip(lower=0)
    numer = sub_sim.values @ user_ratings.values
    denom = sub_sim.values.sum(axis=1)
    return np.where(denom > 0, numer / denom, global_mean)

def recommend_movies(user_id, n=12):
    if user_id not in train_matrix.index:
        return []
    watched_movies = set(train_matrix.loc[user_id].dropna().index)
    unseen_movies = [m for m in train_matrix.columns if m not in watched_movies and m in item_similarity_df.index and m in catalog]
    if not unseen_movies:
        return []
    preds = predict_rating_fast(user_id, unseen_movies)
    top_indices = np.argsort(preds)[::-1][:n]
    results = []
    for idx in top_indices:
        mid = unseen_movies[idx]
        pred_val = float(preds[idx])
        match_pct = min(98, max(75, int((pred_val / 5.0) * 100)))
        results.append({
            "movie_id": mid,
            "predicted_rating": round(pred_val, 2),
            "match_pct": match_pct
        })
    return results

def get_hybrid_recommendations(user_id, movie_id, n=6):
    if movie_id not in item_similarity_df.index or user_id not in train_matrix.index:
        return []
    user_ratings = train_matrix.loc[user_id].dropna()
    watched = set(user_ratings.index)
    
    candidates = [m for m in item_similarity_df.index if m not in watched and m != movie_id and m in catalog]
    if not candidates:
        return []
        
    sim_scores = item_similarity_df.loc[movie_id, candidates].clip(lower=0).fillna(0).values
    preds = predict_rating_fast(user_id, candidates)
    
    hybrid = (0.55 * sim_scores) + (0.45 * (preds / 5.0))
    top_idx = np.argsort(hybrid)[::-1][:n]
    
    results = []
    for i in top_idx:
        mid = candidates[i]
        h_score = float(hybrid[i])
        match_pct = min(99, max(75, int(h_score * 100)))
        results.append({
            "movie_id": mid,
            "hybrid_pct": match_pct,
            "sim_pct": int(float(sim_scores[i]) * 100),
            "pred_rating": round(float(preds[i]), 2)
        })
    return results

MOOD_GENRE_PREFS = {
    "Happy": ["Comedy", "Animation", "Children's", "Musical", "Adventure"],
    "Sad": ["Drama", "War", "Film-Noir"],
    "Excited": ["Action", "Sci-Fi", "Adventure", "Thriller"],
    "Calm": ["Documentary", "Animation", "Fantasy", "Drama"],
    "Romantic": ["Romance", "Comedy", "Drama"],
    "Thrilled": ["Thriller", "Mystery", "Crime", "Horror"]
}

def get_mood_recommendations(mood_name, n=8):
    target_genres = MOOD_GENRE_PREFS.get(mood_name, ["Drama"])
    scored = []
    for mid, m in catalog.items():
        genres = m["genres"]
        overlap = len(set(genres).intersection(target_genres))
        if overlap > 0:
            score = (overlap * 2.0) + (m["avg_rating"] * 1.5) + (np.log1p(m["votes"]) * 0.3)
            scored.append((mid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    base_match = 95
    for i, (mid, _) in enumerate(scored[:n]):
        match_pct = max(76, base_match - (i * 2))
        results.append({
            "movie_id": mid,
            "match_pct": match_pct
        })
    return results

def generate_why_this_movie(movie_id, user_id=1):
    m = catalog.get(movie_id, {})
    genres = m.get("genres", ["Drama"])
    genre_text = " & ".join(genres[:2]) if len(genres) >= 2 else genres[0]
    
    similar = find_similar_movies(movie_id, n=1)
    favorite_anchor = catalog.get(similar[0]["movie_id"])["clean_title"] if similar else "The Matrix"
    
    reasons = [
        f"Similar to your favorite film: <strong>{favorite_anchor}</strong>",
        f"Matches your taste profile for <strong>{genre_text}</strong>",
        f"High critical rating (<strong>{m.get('avg_rating', 4.0)}/5.0</strong>) with <strong>{m.get('approval', 91)}% positive sentiment</strong>",
        f"Popular among users with similar viewing history"
    ]
    return reasons

# 7. Session State
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = 50 # Star Wars
if "watchlist" not in st.session_state:
    st.session_state.watchlist = {50, 100, 127, 1, 56, 313, 98, 174, 318, 181, 22, 64}
if "selected_mood" not in st.session_state:
    st.session_state.selected_mood = "Calm"
if "active_user_id" not in st.session_state:
    st.session_state.active_user_id = 1
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "explore_genre" not in st.session_state:
    st.session_state.explore_genre = "All"
if "hero_index" not in st.session_state:
    st.session_state.hero_index = 0
if "show_notifications" not in st.session_state:
    st.session_state.show_notifications = False

HERO_MOVIES = [50, 100, 127, 313, 56]

# 8. Dark Cinema Integrated Theme CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap');

/* Master Dark Background */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {
    background-color: #0b0d14 !important;
    color: #f8fafc !important;
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
}

/* Hide Streamlit Default Drawer so the Permanent Integrated Sidebar takes full control */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
}

#MainMenu, header, footer, [data-testid="stToolbar"] {
    visibility: hidden;
    height: 0;
}
.main .block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1540px !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* Permanent Integrated Sidebar Rail */
.integrated-sidebar-rail {
    background: #0f121d;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.6rem 1.2rem;
    height: 100%;
    min-height: 820px;
}

/* Inputs & Dropdowns */
.stTextInput input, [data-baseweb="input"] input, [data-baseweb="select"] > div {
    background-color: #151928 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
}
[data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {
    background-color: #151928 !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #f8fafc !important;
}

/* Buttons */
button[kind="primary"] {
    background: linear-gradient(135deg, #e50914 0%, #b80610 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 9px !important;
    box-shadow: 0 4px 15px rgba(229, 9, 20, 0.45) !important;
}
button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
}
button[kind="secondary"]:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
    color: #ffffff !important;
}

/* CineMind Logo */
.brand-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 1.5rem;
}
.brand-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #e50914 0%, #b80610 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 15px rgba(229, 9, 20, 0.45);
}
.brand-text {
    font-family: 'Outfit', sans-serif;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #ffffff;
}
.brand-text span {
    color: #e50914;
}

/* Hero Section */
.hero-banner {
    position: relative;
    border-radius: 18px;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(17, 23, 39, 0.94) 0%, rgba(9, 11, 18, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 3rem 3.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 20px 45px rgba(0, 0, 0, 0.6);
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 46px;
    font-weight: 900;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #ffffff;
    margin-bottom: 0.6rem;
    text-shadow: 0 4px 20px rgba(0,0,0,0.8);
}
.hero-meta {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
    color: #94a3b8;
    margin-bottom: 1.2rem;
    font-weight: 500;
}
.hero-meta .pill {
    background: rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
}
.hero-desc {
    font-size: 15px;
    line-height: 1.6;
    color: #cbd5e1;
    max-width: 680px;
    margin-bottom: 1rem;
}

/* Section Header */
.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 1.5rem 0 1rem 0;
}
.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
}
.see-all-btn {
    font-size: 13px;
    color: #94a3b8;
    font-weight: 600;
}

/* Match pill badge */
.match-badge {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 700;
    display: inline-block;
}

/* Mood Banner */
.mood-header-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #15112e 50%, #0b0d14 100%);
    border: 1px solid rgba(129, 140, 248, 0.2);
    border-radius: 20px;
    padding: 2.4rem 3rem;
    margin-bottom: 2rem;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
}
.mood-header-title {
    font-family: 'Outfit', sans-serif;
    font-size: 34px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.4rem;
}
.mood-header-sub {
    font-size: 15px;
    color: #a5b4fc;
}

/* Why This Movie Card */
.why-movie-card {
    background: rgba(21, 26, 42, 0.95);
    border: 1px solid rgba(229, 9, 20, 0.35);
    border-radius: 16px;
    padding: 1.8rem;
    box-shadow: 0 10px 30px rgba(229, 9, 20, 0.12);
}
.why-title {
    font-family: 'Outfit', sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.3rem;
}
.why-subtitle {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 1.2rem;
}
.reason-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 1.1rem;
    font-size: 14px;
    color: #cbd5e1;
    line-height: 1.5;
}
.reason-icon {
    font-size: 18px;
    flex-shrink: 0;
    margin-top: 1px;
}

/* Notification Drawer */
.notif-box {
    background: rgba(21, 26, 42, 0.98);
    border: 1px solid rgba(229, 9, 20, 0.4);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.6);
}

/* Search Highlight Box */
.search-highlight-card {
    background: linear-gradient(135deg, rgba(229, 9, 20, 0.12) 0%, rgba(21, 26, 42, 0.9) 100%);
    border: 1px solid rgba(229, 9, 20, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 2rem;
}

/* Profile & Stats */
.profile-card {
    background: rgba(18, 23, 38, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.avatar-circle {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: linear-gradient(135deg, #e50914 0%, #4f46e5 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Outfit', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
    margin: 0 auto 1rem auto;
    box-shadow: 0 6px 20px rgba(229, 9, 20, 0.35);
}
.pref-row {
    display: flex;
    justify-content: space-between;
    padding: 0.7rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 13px;
}
.pref-label {
    color: #94a3b8;
}
.pref-val {
    color: #f8fafc;
    font-weight: 600;
}
.glass-panel {
    background: rgba(18, 23, 38, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# 9. Master Two-Column Layout (PERMANENT INTEGRATED SIDEBAR + MAIN CONTENT)
col_permanent_side, col_main_content = st.columns([2.3, 9.7], gap="medium")

# ==========================================
# PERMANENT SIDEBAR RAIL (CAN NEVER DISAPPEAR)
# ==========================================
with col_permanent_side:
    st.markdown("""
    <div class="brand-container">
        <div class="brand-icon">🎬</div>
        <div class="brand-text">Cine<span>Mind</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    pages = [
        ("Home", "🏠  Home"),
        ("Explore", "🔍  Search & Explore"),
        ("Mood", "🎭  Mood Match"),
        ("Movie Details", "🎬  Movie Details"),
        ("Watchlist & Profile", "👤  Watchlist & Profile")
    ]
    
    for page_key, label in pages:
        is_active = (st.session_state.current_page == page_key)
        btn_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"perm_side_{page_key}", type=btn_type, use_container_width=True):
            st.session_state.current_page = page_key
            st.rerun()
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Active MovieLens User Selector
    st.markdown("<div style='font-size:12px; color:#64748b; font-weight:700; text-transform:uppercase; margin-bottom:6px;'>Active User Profile</div>", unsafe_allow_html=True)
    user_ids = sorted(int(u) for u in train_matrix.index)
    selected_uid = st.selectbox(
        "User Profile",
        user_ids,
        index=user_ids.index(st.session_state.active_user_id) if st.session_state.active_user_id in user_ids else 0,
        label_visibility="collapsed",
        key="perm_user_select"
    )
    if selected_uid != st.session_state.active_user_id:
        st.session_state.active_user_id = selected_uid
        st.rerun()
        
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.04); border-radius:12px; padding:12px; margin-top:12px; border:1px solid rgba(255,255,255,0.07);">
        <div style="font-size:11px; color:#94a3b8;">User Profile</div>
        <div style="font-size:13px; font-weight:700; color:#f8fafc; margin-top:2px;">Atharv Shukla (User {st.session_state.active_user_id})</div>
        <div style="font-size:11px; color:#34d399; margin-top:3px;">● Online • Movie Explorer</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# MAIN CONTENT CONTAINER
# ==========================================
with col_main_content:
    # 10. Top Bar (With Functional Search, Notifications, and Profile Click)
    col_search, col_search_btn, col_spacer, col_notif, col_prof = st.columns([6, 1.2, 0.6, 1.2, 2.5])

    with col_search:
        search_input = st.text_input(
            "Search",
            placeholder="Search for movies, actors, genres (e.g. Star Wars, Fargo, Pulp Fiction)...",
            label_visibility="collapsed",
            key="top_search_input"
        )

    with col_search_btn:
        if st.button("🔍 Search", key="top_search_btn", type="primary", use_container_width=True):
            if search_input:
                st.session_state.search_query = search_input
                st.session_state.current_page = "Explore"
                st.rerun()

    with col_notif:
        if st.button("🔔 3", key="top_bell_btn", use_container_width=True):
            st.session_state.show_notifications = not st.session_state.show_notifications
            st.rerun()

    with col_prof:
        if st.button("👤  Atharv Shukla", key="top_profile_btn", use_container_width=True):
            st.session_state.current_page = "Watchlist & Profile"
            st.rerun()

    # Notifications Drawer Dropdown
    if st.session_state.show_notifications:
        st.markdown("""
        <div class="notif-box">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <div style="font-weight:800; font-size:15px; color:#ffffff;">🔔 Notifications (3 New)</div>
                <div style="font-size:12px; color:#94a3b8;">CineMind Intelligence</div>
            </div>
            <div style="font-size:13px; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.06); color:#cbd5e1;">
                ⭐ <strong>Personalized Update:</strong> New recommendations available based on your previous ratings.
            </div>
            <div style="font-size:13px; padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.06); color:#cbd5e1;">
                🎬 <strong>Watchlist Alert:</strong> You have 12 saved movies ready to watch.
            </div>
            <div style="font-size:13px; padding:6px 0; color:#cbd5e1;">
                ⚡ <strong>Trending in Sci-Fi:</strong> High similarity detected with your favorite films.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Close Notifications", key="close_notif_btn"):
            st.session_state.show_notifications = False
            st.rerun()

    st.markdown("<div style='margin-bottom: 1.2rem;'></div>", unsafe_allow_html=True)

    # 11. Page Views

    # ==========================================
    # VIEW 1: HOME DASHBOARD
    # ==========================================
    if st.session_state.current_page == "Home":
        hero_mid = HERO_MOVIES[st.session_state.hero_index % len(HERO_MOVIES)]
        hero_m = catalog[hero_mid]
        
        st.markdown(f"""
        <div class="hero-banner">
            <div style="position:relative; z-index:2;">
                <div class="hero-title">{hero_m['clean_title']}</div>
                <div class="hero-meta">
                    <span>{hero_m['year']}</span> • 
                    <span class="pill">{hero_m['age_rating']}</span> • 
                    <span>{' • '.join(hero_m['genres'][:3])}</span> • 
                    <span>{hero_m['runtime']}</span> • 
                    <span style="color:#fbbf24; font-weight:700;">★ {hero_m['avg_rating']}/5.0</span>
                </div>
                <div class="hero-desc">{hero_m['synopsis']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_play, col_add, col_prev, col_next, _ = st.columns([1.8, 2.0, 0.6, 0.6, 7.0])
        with col_play:
            if st.button("▶  Watch Details", key="hero_play_btn", type="primary", use_container_width=True):
                st.session_state.selected_movie_id = hero_mid
                st.session_state.current_page = "Movie Details"
                st.rerun()
        with col_add:
            in_wl = hero_mid in st.session_state.watchlist
            wl_label = "✓  In Watchlist" if in_wl else "＋  My List"
            if st.button(wl_label, key="hero_wl_btn", use_container_width=True):
                if in_wl:
                    st.session_state.watchlist.remove(hero_mid)
                else:
                    st.session_state.watchlist.add(hero_mid)
                st.rerun()
        with col_prev:
            if st.button("‹", key="hero_prev"):
                st.session_state.hero_index = (st.session_state.hero_index - 1) % len(HERO_MOVIES)
                st.rerun()
        with col_next:
            if st.button("›", key="hero_next"):
                st.session_state.hero_index = (st.session_state.hero_index + 1) % len(HERO_MOVIES)
                st.rerun()

        st.markdown("""
        <div class="section-header">
            <div class="section-title">Recommended for You</div>
            <div class="see-all-btn">See All →</div>
        </div>
        """, unsafe_allow_html=True)
        
        recs = recommend_movies(st.session_state.active_user_id, n=6)
        if not recs:
            recs = [{"movie_id": mid, "match_pct": 92} for mid in [50, 100, 127, 56, 98, 1]]
            
        cols = st.columns(6)
        for i, r in enumerate(recs[:6]):
            mid = r["movie_id"]
            m = catalog.get(mid)
            if not m:
                continue
            with cols[i]:
                svg_data = get_svg_poster(mid)
                st.image(svg_data, use_container_width=True)
                st.markdown(f"""
                <div style="margin-top:6px;">
                    <div style="font-weight:700; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#f8fafc;">{m['clean_title']}</div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2px;">
                        <span style="font-size:12px; color:#94a3b8;">{m['year']}</span>
                        <span style="font-size:12px; color:#34d399; font-weight:700;">{r['match_pct']}% match</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("View", key=f"rec_view_{mid}_{i}", use_container_width=True):
                    st.session_state.selected_movie_id = mid
                    st.session_state.current_page = "Movie Details"
                    st.rerun()

        st.markdown("""
        <div class="section-header" style="margin-top: 2rem;">
            <div class="section-title">Top Rated Classics</div>
            <div class="see-all-btn">Explore Full Library →</div>
        </div>
        """, unsafe_allow_html=True)
        
        top_classics = [318, 64, 127, 50, 174, 98]
        cols_classics = st.columns(6)
        for i, mid in enumerate(top_classics):
            m = catalog.get(mid)
            if not m:
                continue
            with cols_classics[i]:
                svg_data = get_svg_poster(mid)
                st.image(svg_data, use_container_width=True)
                st.markdown(f"""
                <div style="margin-top:6px;">
                    <div style="font-weight:700; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#f8fafc;">{m['clean_title']}</div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2px;">
                        <span style="font-size:12px; color:#94a3b8;">{m['year']}</span>
                        <span style="font-size:12px; color:#fbbf24; font-weight:700;">★ {m['avg_rating']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Inspect", key=f"classic_view_{mid}_{i}", use_container_width=True):
                    st.session_state.selected_movie_id = mid
                    st.session_state.current_page = "Movie Details"
                    st.rerun()

    # ==========================================
    # VIEW 2: MOOD BASED RECOMMENDATIONS
    # ==========================================
    elif st.session_state.current_page == "Mood":
        st.markdown(f"""
        <div class="mood-header-card">
            <div class="mood-header-title">How are you feeling today?</div>
            <div class="mood-header-sub">Tell us your mood and we'll find the perfect movie for you.</div>
        </div>
        """, unsafe_allow_html=True)
        
        moods = [
            ("Happy", "☀️", "Happy", "#fbbf24"),
            ("Sad", "🌧️", "Sad", "#60a5fa"),
            ("Excited", "⚡", "Excited", "#c084fc"),
            ("Calm", "🍃", "Calm", "#34d399"),
            ("Romantic", "💖", "Romantic", "#f472b6"),
            ("Thrilled", "⭐", "Thrilled", "#fb923c"),
        ]
        
        col_m = st.columns(6)
        for i, (m_key, icon, label, color) in enumerate(moods):
            is_selected = (st.session_state.selected_mood == m_key)
            btn_type = "primary" if is_selected else "secondary"
            with col_m[i]:
                if st.button(f"{icon}  {label}", key=f"mood_pick_{m_key}", type=btn_type, use_container_width=True):
                    st.session_state.selected_mood = m_key
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        cur_mood = st.session_state.selected_mood
        cur_icon = dict([(m[0], m[1]) for m in moods]).get(cur_mood, "🍃")
        
        st.markdown(f"""
        <div class="section-header">
            <div class="section-title">Based on your mood: <span style="color:#34d399;">{cur_mood} {cur_icon}</span></div>
            <div class="see-all-btn">See All →</div>
        </div>
        """, unsafe_allow_html=True)
        
        mood_recs = get_mood_recommendations(cur_mood, n=10)
        
        cols_m1 = st.columns(5)
        for i, r in enumerate(mood_recs[:5]):
            mid = r["movie_id"]
            m = catalog.get(mid)
            if not m:
                continue
            with cols_m1[i]:
                svg_data = get_svg_poster(mid)
                st.image(svg_data, use_container_width=True)
                st.markdown(f"""
                <div style="margin-top:6px;">
                    <div style="font-weight:700; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#f8fafc;">{m['clean_title']}</div>
                    <div style="font-size:12px; color:#94a3b8; margin:2px 0;">{' • '.join(m['genres'][:2])}</div>
                    <div class="match-badge">{r['match_pct']}% match</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("View Movie", key=f"m_view_{mid}_{i}", use_container_width=True):
                    st.session_state.selected_movie_id = mid
                    st.session_state.current_page = "Movie Details"
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        cols_m2 = st.columns(5)
        for i, r in enumerate(mood_recs[5:10]):
            mid = r["movie_id"]
            m = catalog.get(mid)
            if not m:
                continue
            with cols_m2[i]:
                svg_data = get_svg_poster(mid)
                st.image(svg_data, use_container_width=True)
                st.markdown(f"""
                <div style="margin-top:6px;">
                    <div style="font-weight:700; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#f8fafc;">{m['clean_title']}</div>
                    <div style="font-size:12px; color:#94a3b8; margin:2px 0;">{' • '.join(m['genres'][:2])}</div>
                    <div class="match-badge">{r['match_pct']}% match</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("View Movie", key=f"m_view_{mid}_{i+5}", use_container_width=True):
                    st.session_state.selected_movie_id = mid
                    st.session_state.current_page = "Movie Details"
                    st.rerun()

    # ==========================================
    # VIEW 3: MOVIE DETAILS (WITH 'WHY THIS MOVIE?')
    # ==========================================
    elif st.session_state.current_page == "Movie Details":
        col_sel, col_empty = st.columns([8, 4])
        with col_sel:
            all_titles = sorted(list(set(m["clean_title"] for m in catalog.values())))
            current_title = catalog[st.session_state.selected_movie_id]["clean_title"]
            picked_title = st.selectbox(
                "Change Selected Movie",
                all_titles,
                index=all_titles.index(current_title) if current_title in all_titles else 0,
                key="details_title_selector"
            )
            if picked_title != current_title:
                for mid, m in catalog.items():
                    if m["clean_title"] == picked_title:
                        st.session_state.selected_movie_id = mid
                        st.rerun()
                        break

        st.markdown("<br>", unsafe_allow_html=True)
        
        mid = st.session_state.selected_movie_id
        m = catalog[mid]
        
        col_poster, col_info, col_why = st.columns([3.5, 4.8, 3.7])
        
        with col_poster:
            svg_poster = get_svg_poster(mid)
            st.image(svg_poster, use_container_width=True)
            
        with col_info:
            st.markdown(f"""
            <div style="font-family:'Outfit', sans-serif; font-size:36px; font-weight:800; color:#ffffff; line-height:1.1; margin-bottom:8px;">
                {m['clean_title']}
            </div>
            <div style="display:flex; align-items:center; gap:10px; font-size:14px; color:#94a3b8; margin-bottom:14px;">
                <span>{m['year']}</span> • 
                <span style="background:rgba(255,255,255,0.1); color:#f8fafc; padding:2px 8px; border-radius:6px; font-size:12px; font-weight:600;">{m['age_rating']}</span> • 
                <span>{' • '.join(m['genres'])}</span>
            </div>
            <div style="display:flex; align-items:center; gap:20px; margin: 1rem 0 1.5rem 0;">
                <div style="font-size:15px; font-weight:700; color:#ffffff;">
                    <span style="color:#fbbf24; font-size:18px;">★</span>
                    <span> {m['star_10']}/10 <span style="font-size:12px; color:#94a3b8; font-weight:400;">({m['votes'] * 3200:,} votes)</span></span>
                </div>
                <div style="font-size:15px; font-weight:700; color:#ffffff;">
                    <span style="font-size:16px;">👍</span>
                    <span> {m['approval']}% <span style="font-size:12px; color:#94a3b8; font-weight:400;">liked this movie</span></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_btn1, col_btn2, col_btn3 = st.columns([1.5, 1.8, 1.2])
            with col_btn1:
                st.button("▶  Play", key="det_play", type="primary", use_container_width=True)
            with col_btn2:
                in_wl = mid in st.session_state.watchlist
                wl_label = "✓  In List" if in_wl else "＋  My List"
                if st.button(wl_label, key="det_add_wl", use_container_width=True):
                    if in_wl:
                        st.session_state.watchlist.remove(mid)
                    else:
                        st.session_state.watchlist.add(mid)
                    st.rerun()
            with col_btn3:
                if st.button("↗  Share", key="det_share", use_container_width=True):
                    st.toast(f"Link copied for {m['clean_title']}!")
                    
            st.markdown(f"""
            <div style="margin-top:1.5rem; font-size:14px; line-height:1.7; color:#cbd5e1;">
                {m['synopsis']}
            </div>
            <div style="margin-top:1.5rem; padding-top:1.2rem; border-top:1px solid rgba(255,255,255,0.08); font-size:13px; line-height:1.8;">
                <div><span style="color:#64748b; font-weight:600;">Director:</span> <span style="color:#f8fafc;">{m['director']}</span></div>
                <div><span style="color:#64748b; font-weight:600;">Cast:</span> <span style="color:#f8fafc;">{m['cast']}</span></div>
                <div><span style="color:#64748b; font-weight:600;">Runtime:</span> <span style="color:#f8fafc;">{m['runtime']}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_why:
            reasons = generate_why_this_movie(mid, st.session_state.active_user_id)
            icons = ["🎬", "🎯", "⭐", "👥"]
            
            st.markdown(f"""
            <div class="why-movie-card">
                <div class="why-title">Why this movie?</div>
                <div class="why-subtitle">Recommended for you because:</div>
                <div class="reason-row">
                    <div class="reason-icon">{icons[0]}</div>
                    <div>{reasons[0]}</div>
                </div>
                <div class="reason-row">
                    <div class="reason-icon">{icons[1]}</div>
                    <div>{reasons[1]}</div>
                </div>
                <div class="reason-row">
                    <div class="reason-icon">{icons[2]}</div>
                    <div>{reasons[2]}</div>
                </div>
                <div class="reason-row">
                    <div class="reason-icon">{icons[3]}</div>
                    <div>{reasons[3]}</div>
                </div>
                <div style="margin-top:1.2rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,0.08); font-size:13px; color:#e50914; font-weight:700;">
                    See similar movies ↓
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-header">
            <div class="section-title">Similar Movies</div>
            <div class="see-all-btn">Explore Graph →</div>
        </div>
        """, unsafe_allow_html=True)
        
        similar_items = find_similar_movies(mid, n=6)
        if similar_items:
            cols_sim = st.columns(6)
            for i, s in enumerate(similar_items):
                smid = s["movie_id"]
                sm = catalog.get(smid)
                if not sm:
                    continue
                with cols_sim[i]:
                    svg_data = get_svg_poster(smid)
                    st.image(svg_data, use_container_width=True)
                    st.markdown(f"""
                    <div style="margin-top:6px;">
                        <div style="font-weight:700; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#f8fafc;">{sm['clean_title']}</div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2px;">
                            <span style="font-size:12px; color:#94a3b8;">{sm['year']}</span>
                            <span style="font-size:12px; color:#34d399; font-weight:700;">{s['match_pct']}% match</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Inspect", key=f"sim_pick_{smid}_{i}", use_container_width=True):
                        st.session_state.selected_movie_id = smid
                        st.rerun()

    # ==========================================
    # VIEW 4: SEARCH & EXPLORE (WITH SEARCH-BASED & PREVIOUS RATINGS RECOMMENDATIONS)
    # ==========================================
    elif st.session_state.current_page == "Explore":
        st.markdown("<div style='font-family:Outfit,sans-serif; font-size:26px; font-weight:800; color:#ffffff; margin-bottom:12px;'>🔍 Search & Intelligent Recommendations</div>", unsafe_allow_html=True)
        
        col_exp_search, col_exp_btn = st.columns([6, 1.5])
        with col_exp_search:
            exp_query = st.text_input(
                "Search Movie",
                value=st.session_state.search_query,
                placeholder="Search movie name (e.g. Star Wars, Fargo, Godfather, Toy Story, Pulp Fiction)...",
                key="explore_search_field"
            )
        with col_exp_btn:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("Find & Recommend", key="explore_submit_btn", type="primary", use_container_width=True):
                st.session_state.search_query = exp_query
                st.rerun()

        st.markdown("<div style='font-size:12px; color:#94a3b8; margin: 4px 0 16px 0;'>Popular Searches: " +
            "<span style='color:#e50914;'>Star Wars</span> • <span style='color:#e50914;'>Fargo</span> • <span style='color:#e50914;'>The Godfather</span> • <span style='color:#e50914;'>Toy Story</span> • <span style='color:#e50914;'>Pulp Fiction</span> • <span style='color:#e50914;'>Titanic</span></div>", unsafe_allow_html=True)
        
        target_movie = None
        if exp_query:
            q_clean = exp_query.strip().lower()
            exact_matches = [m for m in catalog.values() if q_clean == m["clean_title"].lower()]
            if exact_matches:
                target_movie = exact_matches[0]
            else:
                partial_matches = [m for m in catalog.values() if q_clean in m["clean_title"].lower()]
                if partial_matches:
                    target_movie = partial_matches[0]

        if target_movie:
            tmid = target_movie["movie_id"]
            
            st.markdown(f"""
            <div class="search-highlight-card">
                <div style="font-size:13px; color:#e50914; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">🎯 Selected Search Result</div>
                <div style="font-family:Outfit,sans-serif; font-size:28px; font-weight:800; color:#ffffff;">{target_movie['clean_title']} ({target_movie['year']})</div>
                <div style="font-size:14px; color:#94a3b8; margin-top:4px;">
                    {' • '.join(target_movie['genres'])} • Rating: <span style="color:#fbbf24; font-weight:700;">★ {target_movie['avg_rating']}/5.0</span> • Approvals: {target_movie['approval']}%
                </div>
                <div style="font-size:14px; color:#cbd5e1; margin-top:8px; line-height:1.5;">{target_movie['synopsis']}</div>
            </div>
            """, unsafe_allow_html=True)

            col_t_btn1, col_t_btn2, _ = st.columns([2.5, 2.5, 7.0])
            with col_t_btn1:
                if st.button(f"🎬 Full Details on {target_movie['clean_title']}", key="search_open_details", type="primary"):
                    st.session_state.selected_movie_id = tmid
                    st.session_state.current_page = "Movie Details"
                    st.rerun()
            with col_t_btn2:
                in_wl = tmid in st.session_state.watchlist
                wl_txt = "✓ In Watchlist" if in_wl else "＋ Add to Watchlist"
                if st.button(wl_txt, key="search_add_wl"):
                    if in_wl:
                        st.session_state.watchlist.remove(tmid)
                    else:
                        st.session_state.watchlist.add(tmid)
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # 1. HYBRID AI RECOMMENDATIONS (Matches the searched movie + matches user's previous ratings!)
            st.markdown(f"""
            <div class="section-header">
                <div class="section-title">🔮 Smart AI Recommendations <span style="font-size:13px; color:#34d399; font-weight:600;">(Blends '{target_movie['clean_title']}' with your previous rating profile)</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            hybrids = get_hybrid_recommendations(st.session_state.active_user_id, tmid, n=6)
            if hybrids:
                cols_h = st.columns(6)
                for i, h in enumerate(hybrids):
                    hmid = h["movie_id"]
                    hm = catalog.get(hmid)
                    if not hm:
                        continue
                    with cols_h[i]:
                        svg_data = get_svg_poster(hmid)
                        st.image(svg_data, use_container_width=True)
                        st.markdown(f"""
                        <div style="margin-top:6px;">
                            <div style="font-weight:700; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#f8fafc;">{hm['clean_title']}</div>
                            <div style="font-size:12px; color:#94a3b8; margin:2px 0;">{hm['year']} • {hm['primary_genre']}</div>
                            <div class="match-badge" style="font-size:11px;">{h['hybrid_pct']}% AI Match</div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Inspect", key=f"h_pick_{hmid}_{i}", use_container_width=True):
                            st.session_state.selected_movie_id = hmid
                            st.session_state.current_page = "Movie Details"
                            st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # 2. SIMILAR MOVIES BASED ON THIS SEARCHED FILM
            st.markdown(f"""
            <div class="section-header">
                <div class="section-title">🎬 Movies Similar to '{target_movie['clean_title']}' <span style="font-size:13px; color:#94a3b8; font-weight:500;">(Item Cosine Similarity)</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            sim_recs = find_similar_movies(tmid, n=6)
            if sim_recs:
                cols_s = st.columns(6)
                for i, s in enumerate(sim_recs):
                    smid = s["movie_id"]
                    sm = catalog.get(smid)
                    if not sm:
                        continue
                    with cols_s[i]:
                        svg_data = get_svg_poster(smid)
                        st.image(svg_data, use_container_width=True)
                        st.markdown(f"""
                        <div style="margin-top:6px;">
                            <div style="font-weight:700; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#f8fafc;">{sm['clean_title']}</div>
                            <div style="font-size:12px; color:#94a3b8; margin:2px 0;">{sm['year']}</div>
                            <div style="font-size:12px; color:#60a5fa; font-weight:700;">{s['match_pct']}% similarity</div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Inspect", key=f"s_pick_{smid}_{i}", use_container_width=True):
                            st.session_state.selected_movie_id = smid
                            st.session_state.current_page = "Movie Details"
                            st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            # 3. PERSONALIZED RECOMMENDATIONS BASED ON USER'S HISTORICAL RATINGS
            st.markdown(f"""
            <div class="section-header">
                <div class="section-title">⭐ Suggested for You based on Your Previous Ratings <span style="font-size:13px; color:#fbbf24; font-weight:600;">(User Profile {st.session_state.active_user_id})</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            user_recs = recommend_movies(st.session_state.active_user_id, n=6)
            if user_recs:
                cols_u = st.columns(6)
                for i, u in enumerate(user_recs):
                    umid = u["movie_id"]
                    um = catalog.get(umid)
                    if not um:
                        continue
                    with cols_u[i]:
                        svg_data = get_svg_poster(umid)
                        st.image(svg_data, use_container_width=True)
                        st.markdown(f"""
                        <div style="margin-top:6px;">
                            <div style="font-weight:700; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#f8fafc;">{um['clean_title']}</div>
                            <div style="font-size:12px; color:#94a3b8; margin:2px 0;">{um['year']}</div>
                            <div style="font-size:12px; color:#34d399; font-weight:700;">Pred: {u['predicted_rating']} ★</div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Inspect", key=f"u_pick_{umid}_{i}", use_container_width=True):
                            st.session_state.selected_movie_id = umid
                            st.session_state.current_page = "Movie Details"
                            st.rerun()

            st.markdown("<hr style='border-color:rgba(255,255,255,0.08); margin:30px 0;'>", unsafe_allow_html=True)

        # Genre Filter Chips
        st.markdown("<div style='font-size:15px; font-weight:700; color:#ffffff; margin-bottom:8px;'>Browse Catalog by Genre & Filters</div>", unsafe_allow_html=True)
        genre_chips = ["All", "Action", "Drama", "Comedy", "Sci-Fi", "Thriller", "Romance", "Adventure", "Animation", "Crime", "Horror", "Mystery", "War", "Western"]
        
        chip_cols = st.columns(len(genre_chips))
        for i, g in enumerate(genre_chips):
            is_sel = (st.session_state.explore_genre == g)
            btn_type = "primary" if is_sel else "secondary"
            with chip_cols[i]:
                if st.button(g, key=f"chip_{g}", type=btn_type, use_container_width=True):
                    st.session_state.explore_genre = g
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        with fcol1:
            f_year = st.selectbox("Year", ["All", "1990s", "1980s", "1970s", "1960s & Older"], key="filter_year")
        with fcol2:
            f_rating = st.selectbox("Rating", ["All", "4.0+ ★", "3.5+ ★", "3.0+ ★"], key="filter_rating")
        with fcol3:
            f_lang = st.selectbox("Language / Era", ["All", "English", "International Classics"], key="filter_lang")
        with fcol4:
            f_sort = st.selectbox("Sort by", ["Popularity (Votes)", "Highest Rating", "Title (A-Z)", "Release Year"], key="filter_sort")

        filtered = list(catalog.values())
        
        if st.session_state.explore_genre != "All":
            filtered = [m for m in filtered if st.session_state.explore_genre in m["genres"]]
            
        if f_year == "1990s":
            filtered = [m for m in filtered if m["year"].startswith("199")]
        elif f_year == "1980s":
            filtered = [m for m in filtered if m["year"].startswith("198")]
        elif f_year == "1970s":
            filtered = [m for m in filtered if m["year"].startswith("197")]
        elif f_year == "1960s & Older":
            filtered = [m for m in filtered if int(m["year"]) <= 1969]
            
        if f_rating == "4.0+ ★":
            filtered = [m for m in filtered if m["avg_rating"] >= 4.0]
        elif f_rating == "3.5+ ★":
            filtered = [m for m in filtered if m["avg_rating"] >= 3.5]
        elif f_rating == "3.0+ ★":
            filtered = [m for m in filtered if m["avg_rating"] >= 3.0]
            
        if f_sort == "Popularity (Votes)":
            filtered.sort(key=lambda x: x["votes"], reverse=True)
        elif f_sort == "Highest Rating":
            filtered.sort(key=lambda x: (x["avg_rating"], x["votes"]), reverse=True)
        elif f_sort == "Title (A-Z)":
            filtered.sort(key=lambda x: x["clean_title"])
        elif f_sort == "Release Year":
            filtered.sort(key=lambda x: x["year"], reverse=True)

        st.markdown(f"<div style='font-size:14px; color:#94a3b8; margin: 10px 0 20px 0;'>Showing <strong style='color:#ffffff;'>{len(filtered)}</strong> matching movies in library</div>", unsafe_allow_html=True)
        
        grid_size = 15
        display_items = filtered[:grid_size]
        
        for row_start in range(0, len(display_items), 5):
            row_items = display_items[row_start : row_start + 5]
            cols_grid = st.columns(5)
            for i, m in enumerate(row_items):
                mid = m["movie_id"]
                with cols_grid[i]:
                    svg_data = get_svg_poster(mid)
                    st.image(svg_data, use_container_width=True)
                    st.markdown(f"""
                    <div style="margin-top:6px;">
                        <div style="font-weight:700; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#f8fafc;">{m['clean_title']}</div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2px;">
                            <span style="font-size:12px; color:#94a3b8;">{m['year']}</span>
                            <span style="font-size:12px; color:#fbbf24; font-weight:700;">★ {m['avg_rating']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Details", key=f"grid_view_{mid}_{row_start+i}", use_container_width=True):
                        st.session_state.selected_movie_id = mid
                        st.session_state.current_page = "Movie Details"
                        st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # VIEW 5: WATCHLIST & PROFILE (USER PREFERENCES & STATS)
    # ==========================================
    elif st.session_state.current_page == "Watchlist & Profile":
        col_profile, col_watchlist, col_picks = st.columns([3.2, 5.2, 3.6])
        
        with col_profile:
            st.markdown("""
            <div class="profile-card">
                <div class="avatar-circle">AS</div>
                <div style="font-family:'Outfit',sans-serif; font-size:22px; font-weight:800; color:#ffffff;">Atharv Shukla</div>
                <div style="font-size:13px; color:#94a3b8; margin-bottom:1rem;">Movie Explorer</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("✎  Edit Profile", key="edit_prof_btn", use_container_width=True):
                st.toast("Profile settings are active for Atharv Shukla!")
            
            st.markdown("""
            <div class="glass-panel" style="margin-top:1.5rem;">
                <div style="font-family:'Outfit',sans-serif; font-size:16px; font-weight:700; color:#ffffff; margin-bottom:1rem;">Your Preferences</div>
                <div class="pref-row">
                    <span class="pref-label">🎬 Favorite Genres</span>
                    <span class="pref-val">Sci-Fi, Thriller, Drama</span>
                </div>
                <div class="pref-row">
                    <span class="pref-label">☁️ Preferred Mood</span>
                    <span class="pref-val">Calm, Thoughtful, Excited</span>
                </div>
                <div class="pref-row">
                    <span class="pref-label">🌐 Languages</span>
                    <span class="pref-val">English, Hindi</span>
                </div>
                <div class="pref-row" style="border-bottom:none;">
                    <span class="pref-label">🔞 Content Rating</span>
                    <span class="pref-val">13+</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_watchlist:
            wl_list = list(st.session_state.watchlist)
            st.markdown(f"""
            <div class="section-header">
                <div class="section-title">My Watchlist <span style="font-size:13px; color:#94a3b8; font-weight:500;">({len(wl_list)} movies)</span></div>
                <div class="see-all-btn">See All →</div>
            </div>
            """, unsafe_allow_html=True)
            
            if wl_list:
                wl_cols = st.columns(4)
                for i, wmid in enumerate(wl_list[:4]):
                    wm = catalog.get(wmid)
                    if not wm:
                        continue
                    with wl_cols[i]:
                        svg_data = get_svg_poster(wmid)
                        st.image(svg_data, use_container_width=True)
                        st.markdown(f"""
                        <div style="margin-top:4px; font-size:12px; font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#f8fafc;">{wm['clean_title']}</div>
                        <div style="font-size:11px; color:#94a3b8;">{wm['year']}</div>
                        """, unsafe_allow_html=True)
                        if st.button("Remove", key=f"wl_rm_{wmid}", use_container_width=True):
                            st.session_state.watchlist.remove(wmid)
                            st.rerun()
            else:
                st.info("Your watchlist is currently empty. Add movies using the '＋ My List' buttons.")
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            uid = st.session_state.active_user_id
            watched_count = int(train_matrix.loc[uid].count()) if uid in train_matrix.index else 48
            
            st.markdown(f"""
            <div class="section-title" style="margin-bottom:12px;">Your Movie Stats</div>
            <div style="display:flex; gap:14px; margin-bottom:1.5rem;">
                <div style="flex:1; background:rgba(255,255,255,0.04); border-radius:12px; padding:16px; text-align:center; border:1px solid rgba(255,255,255,0.08);">
                    <div style="font-size:26px; font-weight:800; color:#ffffff;">{watched_count}</div>
                    <div style="font-size:12px; color:#94a3b8;">Movies Watched</div>
                </div>
                <div style="flex:1; background:rgba(255,255,255,0.04); border-radius:12px; padding:16px; text-align:center; border:1px solid rgba(255,255,255,0.08);">
                    <div style="font-size:26px; font-weight:800; color:#ffffff;">{len(wl_list)}</div>
                    <div style="font-size:12px; color:#94a3b8;">In Watchlist</div>
                </div>
                <div style="flex:1; background:rgba(255,255,255,0.04); border-radius:12px; padding:16px; text-align:center; border:1px solid rgba(255,255,255,0.08);">
                    <div style="font-size:26px; font-weight:800; color:#ffffff;">6</div>
                    <div style="font-size:12px; color:#94a3b8;">Favorite Genres</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="glass-panel">
                <div style="font-family:'Outfit',sans-serif; font-size:16px; font-weight:700; color:#ffffff; margin-bottom:14px;">Your Top Genres</div>
                <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                        <span style="color:#00f2fe; font-weight:600;">Sci-Fi</span>
                        <span style="color:#94a3b8;">45%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.1); border-radius:6px; height:6px; overflow:hidden;">
                        <div style="background:#00f2fe; width:45%; height:100%;"></div>
                    </div>
                </div>
                <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                        <span style="color:#c084fc; font-weight:600;">Thriller</span>
                        <span style="color:#94a3b8;">30%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.1); border-radius:6px; height:6px; overflow:hidden;">
                        <div style="background:#c084fc; width:30%; height:100%;"></div>
                    </div>
                </div>
                <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                        <span style="color:#f472b6; font-weight:600;">Drama</span>
                        <span style="color:#94a3b8;">25%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.1); border-radius:6px; height:6px; overflow:hidden;">
                        <div style="background:#f472b6; width:25%; height:100%;"></div>
                    </div>
                </div>
                <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                        <span style="color:#ef4444; font-weight:600;">Action</span>
                        <span style="color:#94a3b8;">20%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.1); border-radius:6px; height:6px; overflow:hidden;">
                        <div style="background:#ef4444; width:20%; height:100%;"></div>
                    </div>
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                        <span style="color:#fbbf24; font-weight:600;">Comedy</span>
                        <span style="color:#94a3b8;">15%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.1); border-radius:6px; height:6px; overflow:hidden;">
                        <div style="background:#fbbf24; width:15%; height:100%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_picks:
            st.markdown("""
            <div class="section-header">
                <div class="section-title">Top Picks Just for You</div>
                <div class="see-all-btn">🔄</div>
            </div>
            """, unsafe_allow_html=True)
            
            top_picks = recommend_movies(st.session_state.active_user_id, n=4)
            if not top_picks:
                top_picks = [{"movie_id": mid, "match_pct": 90} for mid in [50, 100, 127, 56]]
                
            for i, p in enumerate(top_picks):
                pmid = p["movie_id"]
                pm = catalog.get(pmid)
                if not pm:
                    continue
                with st.container():
                    c_p_img, c_p_txt = st.columns([1, 2])
                    with c_p_img:
                        svg_data = get_svg_poster(pmid)
                        st.image(svg_data, use_container_width=True)
                    with c_p_txt:
                        st.markdown(f"""
                        <div style="font-weight:700; font-size:14px; margin-bottom:2px; color:#ffffff;">{pm['clean_title']}</div>
                        <div style="font-size:12px; color:#94a3b8; margin-bottom:6px;">{pm['year']}</div>
                        <div class="match-badge" style="margin-bottom:8px;">{p['match_pct']}% match</div>
                        """, unsafe_allow_html=True)
                        in_wl = pmid in st.session_state.watchlist
                        wl_btn_txt = "✓ In List" if in_wl else "＋ Add"
                        if st.button(wl_btn_txt, key=f"top_pick_add_{pmid}_{i}", use_container_width=True):
                            if in_wl:
                                st.session_state.watchlist.remove(pmid)
                            else:
                                st.session_state.watchlist.add(pmid)
                            st.rerun()
                    st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:12px 0;'>", unsafe_allow_html=True)
