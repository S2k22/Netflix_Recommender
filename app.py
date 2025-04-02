import streamlit as st
import pandas as pd
import json
import os
import requests
from io import StringIO
import gc
import time
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('netflix_app')

# Log startup information
logger.info("Starting Netflix Recommender App...")

# =============== Initialize Session State Keys ===============
if "loading" not in st.session_state:
    st.session_state.loading = True
if "model_built" not in st.session_state:
    st.session_state.model_built = False
if "selected_titles" not in st.session_state:
    st.session_state.selected_titles = []
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = None
if "recommendation_mode" not in st.session_state:
    st.session_state.recommendation_mode = False
if "show_detail_modal" not in st.session_state:
    st.session_state.show_detail_modal = False
if "detail_title_id" not in st.session_state:
    st.session_state.detail_title_id = None
if "error_count" not in st.session_state:
    st.session_state.error_count = 0
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# Set page configuration
st.set_page_config(
    page_title="Netflix Recommender System",
    page_icon="🎬",
    layout="wide"
)

# ===================== Custom CSS =====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #E50914;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #221f1f;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    .recommendation {
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 5px;
        background-color: #f1f1f1;
        transition: transform 0.2s;
    }
    .recommendation:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    .score-badge {
        background-color: #E50914;
        color: white;
        border-radius: 20px;
        padding: 0.2rem 0.6rem;
        font-weight: bold;
    }
    .combined-score-badge {
        background-color: #221f1f;
        color: white;
        border-radius: 20px;
        padding: 0.2rem 0.6rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .footer {
        text-align: center;
        color: #777;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
    .genre-pill {
        display: inline-block;
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        margin: 0.3rem;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .genre-pill:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
    }
    .genre-pill.active {
        background-color: #E50914;
        color: white;
        border-color: #E50914;
    }
    .tab-content {
        padding: 1rem;
        border: 1px solid #dee2e6;
        border-top: 0;
        border-radius: 0 0 0.25rem 0.25rem;
    }
    .recommendation-type-selector {
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    .explanation-box {
        background-color: #f8f9fa;
        border-left: 4px solid #E50914;
        padding: 0.5rem 1rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .detail-modal {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }
    .truncate-text {
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .system-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ===================== Data & Model Loading =====================
GITHUB_TITLES_URL = "https://raw.githubusercontent.com/S2k22/Netflix_Recommender/master/titles.csv"
GITHUB_CREDITS_URL = "https://raw.githubusercontent.com/S2k22/Netflix_Recommender/master/credits.csv"


@st.cache_data(ttl=3600)  # Cache for 1 hour
def calculate_combined_score(row):
    imdb_score = row['imdb_score'] if pd.notna(row['imdb_score']) else 0
    imdb_votes = row['imdb_votes'] if pd.notna(row['imdb_votes']) else 0
    tmdb_popularity = row['tmdb_popularity'] if pd.notna(row['tmdb_popularity']) else 0
    tmdb_score = row['tmdb_score'] if pd.notna(row['tmdb_score']) else 0
    combined_score = (
            0.4 * imdb_score +
            0.3 * (imdb_votes / 100000) +
            0.2 * tmdb_score +
            0.1 * (tmdb_popularity / 100)
    )
    return combined_score


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data_from_github(url):
    try:
        logger.info(f"Loading data from: {url}")
        response = requests.get(url, timeout=30)  # 30 second timeout
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None


# Simplified recommendation function that uses basic filtering
def get_simple_recommendations(titles_df, liked_ids, num_recommendations=10, genre_filter=None):
    # Filter out already liked titles
    filtered_df = titles_df[~titles_df['id'].isin(liked_ids)].copy()

    # If genre filter is specified, apply it
    if genre_filter:
        filtered_df = filtered_df[filtered_df['genres'].apply(
            lambda x: isinstance(x, list) and genre_filter in x or
                      (isinstance(x, str) and genre_filter in x)
        )]

    # Sort by combined score and return top recommendations
    recommendations = []
    for _, row in filtered_df.sort_values('combined_score', ascending=False).head(num_recommendations).iterrows():
        recommendations.append({
            'id': row['id'],
            'title': row['title'],
            'score': row['combined_score']
        })

    return recommendations


# Get popular titles in a specific genre
def get_popular_in_genre(titles_df, genre, top_n=10):
    genre_matches = []
    for _, row in titles_df.iterrows():
        genres = row.get('genres', [])
        if isinstance(genres, list) and genre in genres:
            genre_matches.append({
                "id": row["id"],
                "title": row["title"],
                "combined_score": row.get("combined_score", 0)
            })
        elif isinstance(genres, str) and genre in genres:
            genre_matches.append({
                "id": row["id"],
                "title": row["title"],
                "combined_score": row.get("combined_score", 0)
            })

    return sorted(genre_matches, key=lambda x: x.get("combined_score", 0), reverse=True)[:top_n]


def load_data():
    try:
        # Load data
        titles_df = load_data_from_github(GITHUB_TITLES_URL)
        credits_df = load_data_from_github(GITHUB_CREDITS_URL)

        if titles_df is None or credits_df is None:
            return None, None

        # Convert numeric columns
        numeric_columns = ['imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']
        for col in numeric_columns:
            if col in titles_df.columns:
                titles_df[col] = pd.to_numeric(titles_df[col], errors='coerce')

        # Handle release_year and runtime
        if 'release_year' in titles_df.columns:
            titles_df['release_year'] = pd.to_numeric(titles_df['release_year'], errors='coerce')
        if 'runtime' in titles_df.columns:
            titles_df['runtime'] = pd.to_numeric(titles_df['runtime'], errors='coerce')

        # Calculate combined score
        titles_df['combined_score'] = titles_df.apply(calculate_combined_score, axis=1)

        # Process genres
        def parse_genres(genres):
            if pd.isna(genres) or genres == '':
                return []
            if isinstance(genres, list):
                return genres
            try:
                return json.loads(genres.replace("'", '"'))
            except:
                return [genres]

        titles_df['genres'] = titles_df['genres'].apply(parse_genres)

        # Sample data if too large
        max_titles = 2000  # Reduced to 2000 to improve performance
        if len(titles_df) > max_titles:
            titles_df = titles_df.sort_values('combined_score', ascending=False).head(max_titles)

        return titles_df, credits_df

    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None, None


# ===================== MAIN APP LAYOUT =====================
try:
    # Show header
    st.markdown("<h1 class='main-header'>Netflix Simple Recommender System</h1>", unsafe_allow_html=True)

    # ======== SIDEBAR ========
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/7a/Logonetflix.png", width=200)
        st.markdown("### Recommendation Settings")

        recommendation_type = st.radio(
            "Select how recommendations are generated:",
            ["Popularity Based", "Genre Based"],
            key="recommendation_type_radio"
        )

        num_recommendations = st.slider("Number of recommendations", 5, 20, 10)

        st.markdown("### About")
        st.markdown("""
        This simplified recommender system provides recommendations based on Netflix content popularity.

        The dataset includes thousands of movies and TV shows with their ratings and metadata.
        """)

    # Main content area
    if "titles_df" not in st.session_state or "credits_df" not in st.session_state:
        with st.spinner("Loading dataset..."):
            titles_df, credits_df = load_data()
            if titles_df is not None and credits_df is not None:
                st.session_state.titles_df = titles_df
                st.session_state.credits_df = credits_df
                st.session_state.model_built = True
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to load data. Please refresh the page and try again.")
                st.stop()
    else:
        titles_df = st.session_state.titles_df
        credits_df = st.session_state.credits_df

    # Are we in "recommendation mode" or "selection mode"?
    if not st.session_state.recommendation_mode:
        # ======== SELECTION MODE ========
        tab1, tab2 = st.tabs(["Top Rated Titles", "Popular by Genre"])

        with tab1:
            st.markdown("<h2 class='sub-header'>Select Titles You Like</h2>", unsafe_allow_html=True)
            st.markdown("Choose a few titles you enjoy, and we'll recommend similar content you might like.")

            if not titles_df.empty:
                top_titles = titles_df.sort_values('combined_score', ascending=False).head(20)
            else:
                st.error("No titles available to display!")
                top_titles = pd.DataFrame()

            cols_per_row = 3
            for i in range(0, len(top_titles), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(top_titles):
                        title = top_titles.iloc[idx]
                        with cols[j]:
                            with st.container():
                                st.markdown(f"### {title['title']}")
                                st.markdown(f"**{title.get('type', 'N/A')} ({title.get('release_year', 'N/A')})**")

                                genres = title.get('genres', [])
                                if len(genres) > 0:
                                    st.markdown(f"**Genres:** {', '.join(genres[:3])}")

                                description = title.get('description', '')
                                if description:
                                    st.markdown(f"<div class='truncate-text'>{description}</div>",
                                                unsafe_allow_html=True)

                                imdb_score = title['imdb_score'] if pd.notna(title['imdb_score']) else "N/A"
                                combined_score = title['combined_score'] if pd.notna(title['combined_score']) else "N/A"
                                combined_score_formatted = f"{combined_score:.2f}" if isinstance(combined_score,
                                                                                                 float) else combined_score
                                st.markdown(
                                    f"**IMDB Score:** {imdb_score} "
                                    f"<span class='combined-score-badge'>Combined Score: {combined_score_formatted}</span>",
                                    unsafe_allow_html=True
                                )

                                title_id = title['id']
                                title_name = title['title']
                                is_selected = any(item['id'] == title_id for item in st.session_state.selected_titles)

                                if is_selected:
                                    if st.button("Deselect", key=f"deselect_{title_id}"):
                                        st.session_state.selected_titles = [
                                            item for item in st.session_state.selected_titles if item['id'] != title_id
                                        ]
                                        st.rerun()
                                else:
                                    if st.button("Select", key=f"select_{title_id}"):
                                        st.session_state.selected_titles.append({
                                            'id': title_id,
                                            'title': title_name
                                        })
                                        st.rerun()

        with tab2:
            st.markdown("<h2 class='sub-header'>Browse Popular Titles by Genre</h2>", unsafe_allow_html=True)
            st.markdown("Explore top-rated titles in specific genres and add them to your selection.")

            # Extract all unique genres
            all_genres = []
            for genres_list in titles_df['genres']:
                if isinstance(genres_list, list):
                    all_genres.extend(genres_list)

            unique_genres = sorted(list(set(all_genres)))
            st.markdown("### Select a Genre")

            genres_per_row = 5
            genre_rows = [unique_genres[i:i + genres_per_row] for i in range(0, len(unique_genres), genres_per_row)]
            for row in genre_rows:
                cols = st.columns(genres_per_row)
                for i, genre in enumerate(row):
                    with cols[i]:
                        if st.button(genre, key=f"genre_{genre}"):
                            st.session_state.selected_genre = genre
                            st.rerun()

            if st.session_state.selected_genre:
                st.markdown(f"### Popular Titles in {st.session_state.selected_genre}")
                popular_in_genre = get_popular_in_genre(titles_df, st.session_state.selected_genre, top_n=12)

                if popular_in_genre:
                    cols_per_row = 3
                    for i in range(0, len(popular_in_genre), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            idx = i + j
                            if idx < len(popular_in_genre):
                                rec = popular_in_genre[idx]
                                title_info = titles_df[titles_df['id'] == rec['id']]
                                if not title_info.empty:
                                    title_details = title_info.iloc[0]
                                    with cols[j]:
                                        with st.container():
                                            st.markdown(f"### {rec['title']}")
                                            st.markdown(
                                                f"**{title_details.get('type', 'N/A')} ({title_details.get('release_year', 'N/A')})**")

                                            genres_val = title_details.get('genres', [])
                                            if len(genres_val) > 0:
                                                st.markdown(f"**Genres:** {', '.join(genres_val[:3])}")

                                            description = title_details.get('description', '')
                                            if description:
                                                st.markdown(f"<div class='truncate-text'>{description}</div>",
                                                            unsafe_allow_html=True)

                                            title_id = rec['id']
                                            title_name = rec['title']
                                            is_selected = any(
                                                item['id'] == title_id for item in st.session_state.selected_titles)

                                            if is_selected:
                                                if st.button("Deselect", key=f"genre_deselect_{title_id}"):
                                                    st.session_state.selected_titles = [
                                                        item for item in st.session_state.selected_titles if
                                                        item['id'] != title_id
                                                    ]
                                                    st.rerun()
                                            else:
                                                if st.button("Select", key=f"genre_select_{title_id}"):
                                                    st.session_state.selected_titles.append({
                                                        'id': title_id,
                                                        'title': title_name
                                                    })
                                                    st.rerun()
                else:
                    st.warning(f"No titles found in the {st.session_state.selected_genre} genre.")

        # Show the "Get Recommendations" button outside the tabs
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Your Selections")
        if st.session_state.selected_titles:
            st.markdown("You've selected:")
            for i, item in enumerate(st.session_state.selected_titles):
                st.markdown(f"{i + 1}. {item['title']}")
            # The button for final recommendations:
            if st.button("Get Recommendations", key="get_recs_button"):
                st.session_state.recommendation_mode = True
                st.rerun()
        else:
            st.markdown("You haven't selected any titles yet. Please select at least one title to get recommendations.")
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # ======== RECOMMENDATION MODE ========
        st.markdown("<h2 class='sub-header'>Your Personalized Recommendations</h2>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Based on your selections:")
        for i, item in enumerate(st.session_state.selected_titles):
            st.markdown(f"{i + 1}. {item['title']}")
        if st.button("Start Over"):
            st.session_state.recommendation_mode = False
            st.session_state.selected_titles = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        liked_ids = [item['id'] for item in st.session_state.selected_titles]

        st.markdown("<div class='recommendation-type-selector'>", unsafe_allow_html=True)
        if recommendation_type == "Genre Based":
            st.markdown("🎭 **Using Genre Based recommendations** - focusing on genres you might like")

            # Get all genres from liked titles
            liked_genres = []
            for title_id in liked_ids:
                title_info = titles_df[titles_df['id'] == title_id]
                if not title_info.empty:
                    genres = title_info.iloc[0].get('genres', [])
                    if isinstance(genres, list):
                        liked_genres.extend(genres)

            # Get recommendations for each genre
            all_recommendations = []
            for genre in set(liked_genres):
                all_recommendations.extend(get_popular_in_genre(titles_df, genre, top_n=5))

            # Remove duplicates and already liked titles
            recommendations = []
            seen_ids = set()
            for rec in all_recommendations:
                if rec['id'] not in liked_ids and rec['id'] not in seen_ids:
                    recommendations.append(rec)
                    seen_ids.add(rec['id'])
                    if len(recommendations) >= num_recommendations:
                        break
        else:
            st.markdown("🔍 **Using Popularity Based recommendations** - focusing on what others enjoyed")
            recommendations = get_simple_recommendations(titles_df, liked_ids, num_recommendations)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<h3 class='sub-header'>Recommendations For You</h3>", unsafe_allow_html=True)
        if not recommendations:
            st.warning("No recommendations found. Try selecting different titles or a different recommendation type.")
        else:
            cols_per_row = 3
            for i in range(0, len(recommendations), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i + j
                    if idx < len(recommendations):
                        rec = recommendations[idx]
                        rec_info = titles_df[titles_df['id'] == rec['id']]
                        if not rec_info.empty:
                            rec_details = rec_info.iloc[0]
                            with cols[j]:
                                with st.container():
                                    st.markdown(f"### {idx + 1}. {rec['title']}")
                                    st.markdown(
                                        f"**{rec_details.get('type', 'N/A')} ({rec_details.get('release_year', 'N/A')})**")

                                    genres_val = rec_details.get('genres', [])
                                    if len(genres_val) > 0:
                                        st.markdown(f"**Genres:** {', '.join(genres_val[:3])}")

                                    description = rec_details.get('description', '')
                                    if description:
                                        st.markdown(f"<div class='truncate-text'>{description}</div>",
                                                    unsafe_allow_html=True)

                                    score = rec.get('combined_score', rec.get('score', 0))
                                    score_text = f"<span class='score-badge'>Score: {score:.2f}</span>"
                                    st.markdown(f"{score_text}", unsafe_allow_html=True)

    # Footer
    st.markdown("<div class='footer'>Netflix Simple Recommender System - Created with Streamlit</div>",
                unsafe_allow_html=True)

except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")

    st.markdown("<div class='system-message'>", unsafe_allow_html=True)
    st.markdown("### System Error")
    st.markdown(
        "The application has encountered an unexpected error. This might be due to memory issues or a temporary problem.")

    if st.button("Reload Page"):
        st.markdown(
            """
            <script>
                window.location.reload();
            </script>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)