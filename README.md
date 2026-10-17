<div align="center">

# 🎬 CineMind — AI-Powered Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![MovieLens](https://img.shields.io/badge/Dataset-MovieLens%20100K-blue?style=for-the-badge)](https://grouplens.org/datasets/movielens/100k/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**An intelligent, multi-tier cinematic movie recommender combining item-based collaborative filtering, cosine similarity, mood matching, and an immersive dark-mode UI.**

[<u><font color="#0969da"><b>🌐 Live Demo</b></font></u>](https://movierecommend-g4ii2wahfhdub3cp6ih5ql.streamlit.app/) • [Key Features](#-key-features) • [Algorithm & Math](#-recommendation-methodology) • [Installation](#-installation--setup) • [Project Structure](#-project-structure)

</div>

---

## 📌 Overview

> 🚀 **Live Demo:** Experience the deployed application live at: [<u><font color="#0969da"><b>https://movierecommend-g4ii2wahfhdub3cp6ih5ql.streamlit.app/</b></font></u>](https://movierecommend-g4ii2wahfhdub3cp6ih5ql.streamlit.app/)

**CineMind** is an end-to-end movie recommendation web application built to eliminate decision fatigue in digital streaming. Powered by the **MovieLens 100K dataset** (1,682 films, 943 users, and 100,000 ratings), CineMind combines **Item-Based Collaborative Filtering**, **Cosine Item Similarity**, and a **Smart Hybrid Fusion Algorithm** to generate real-time (<10ms) personalized recommendations.

The frontend is crafted using Streamlit with a custom obsidian dark-mode design system, glassmorphism containers, a permanent navigation rail, and an offline SVG dynamic poster generator.

---

## ✨ Key Features

### 1. 🏠 Home Dashboard & Cinematic Hero
- **Dynamic Spotlight Carousel**: Features iconic cinema (e.g., *Star Wars*) with year, content rating, genres, runtime, community ratings, and synopsis.
- **Personalized Recommendations**: Instant collaborative filtering suggestions tailored to the active user profile.
- **Top Rated Classics**: High-scoring community favorites with rating badges and direct watchlist triggers.

### 2. 🔍 Multi-Tiered Search & Intelligent Recommendations
When searching for any title (e.g., *Star Wars*, *Fargo*, *Pulp Fiction*), CineMind provides a 4-tier discovery view:
- **Search Result Spotlight**: Detailed movie breakdown.
- **Smart AI Recommendations**: A hybrid scoring engine blending the searched movie's vector similarity with the user's historical ratings.
- **Similar Movies**: Direct item-to-item cosine similarity.
- **Suggested for You**: Profile-specific collaborative filtering recommendations.

### 3. 🎭 Mood Match Discovery
Sentiment-driven film discovery matching viewer emotion to curated genre sets:
- **😌 Chill & Unwind**: Comedy, Animation, Romance, Children's
- **🤯 Mind-Bending**: Sci-Fi, Mystery, Thriller, Film-Noir
- **❤️ Romantic & Heartfelt**: Romance, Drama, Musical
- **⚡ Adrenaline Rush**: Action, Adventure, Crime, War
- **✨ Feel-Good & Inspiring**: Drama, Comedy, Musical, Animation

### 4. 🎬 Movie Details & "Why This Movie?" Explainability
- Complete metadata: release year, runtime, full genre taxonomy, community rating, approval percentage.
- **AI Recommendation Rationale**: Explainable recommendations detailing why the film matches the user's viewing history.
- **Similar Cinema Carousel**: Instant item-similarity suggestions related to the inspected title.

### 5. 👤 Personalized Watchlist & Profile Management
- **Atharv Shukla Profile**: Dedicated profile view with interactive statistics (movies watched, saved watchlist count, favorite genres).
- **Interactive Watchlist**: Add/remove movies across any screen with real-time state persistence.
- **Active User Switcher**: Toggle between 943 real MovieLens user profiles to test recommendation behavior across distinct user taste clusters.

### 6. 🎨 Offline SVG Dynamic Poster Generator
- Autonomous vector graphic poster generator built directly into Python.
- Maps 18 MovieLens genres to custom color gradients, mood motifs, and rating plaques.
- **100% reliable**: Zero external API dependencies, zero rate limiting, zero broken image hotlinks.

---

## 📐 Recommendation Methodology

### 1. Item-Based Collaborative Filtering (IBCF)
Predicted rating $\hat{r}_{u,i}$ for user $u$ on unrated item $i$ is calculated using the weighted sum of ratings user $u$ gave to similar items $j$:

$$\hat{r}_{u,i} = \bar{r} + \frac{\sum_{j \in \text{Rated}(u)} \text{sim}(i, j) \cdot (r_{u,j} - \bar{r})}{\sum_{j \in \text{Rated}(u)} |\text{sim}(i, j)|}$$

Where:
- $\bar{r}$: Global average rating across all interactions ($\approx 3.53$).
- $\text{sim}(i, j)$: Cosine similarity between item feature vectors $i$ and $j$.
- $r_{u,j}$: Actual rating given by user $u$ to item $j$.

### 2. Hybrid Search Scoring Formula
To balance topical relevance of a searched film with long-term user taste:

$$\text{HybridScore}(i) = 0.55 \cdot \text{CosineSimilarity}(i, \text{searched}) + 0.45 \cdot \left(\frac{\hat{r}_{u,i}}{5.0}\right)$$

---

## 📁 Project Structure

```bash
Movie_Recommend/
├── app.py                 # Core Streamlit application & cinematic UI
├── movie_info.csv         # Catalog of 1,682 movies (titles, release dates, URLs)
├── train_matrix.pkl       # Sparse User-Item interaction matrix (943 users × 1,653 items)
├── item_similarity.pkl    # Precomputed Cosine Similarity matrix (1,653 × 1,653)
├── global_mean.pkl        # Baseline global mean rating scalar (3.53)
├── requirements.txt       # Python environment dependencies
└── README.md              # Project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.10+** (Python 3.10, 3.11, 3.12, 3.13, 3.14 supported)
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/atharvshukla76/Movie_Recommend.git
cd Movie_Recommend
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

The application will launch in your browser at:
```
http://localhost:8501/
```

---

## 🛠️ Technology Stack

| Domain | Technology |
|---|---|
| **Language** | Python 3 |
| **Web Framework** | Streamlit |
| **Machine Learning** | Scikit-Learn, Item-Based Collaborative Filtering, Cosine Similarity |
| **Data Processing** | Pandas, NumPy, Joblib |
| **Styling & UI** | Vanilla CSS3 (Obsidian Cinema theme, Glassmorphism, Google Fonts: Plus Jakarta Sans & Outfit) |
| **Asset Engine** | Autonomous SVG vector poster generator (pure Python) |
| **Dataset** | MovieLens 100K Benchmark |

---

## 📊 Dataset Specifications

- **Dataset**: GroupLens MovieLens 100K
- **Users**: 943 unique viewers
- **Movies**: 1,682 titles
- **Ratings**: 100,000 interactions (scale 1–5)
- **Genres (18 categories)**: Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western.

---

## 👤 Author

**Atharv Shukla**
- GitHub: [@atharvshukla76](https://github.com/atharvshukla76)
- Repository: [Movie_Recommend](https://github.com/atharvshukla76/Movie_Recommend)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
