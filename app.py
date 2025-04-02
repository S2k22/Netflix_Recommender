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
if "loading_message" not in st.session_state:
    st.session_state.loading_message = "Loading dataset and building recommendation models..."
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

# ===================== NLTK Setup =====================
try:
    import nltk

    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    nltk.data.path.append(nltk_data_dir)

    # Simple downloads with less output
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)

    logger.info("NLTK resources loaded successfully")
except Exception as e:
    logger.warning(f"NLTK setup issue: {str(e)}")
    st.warning("Some text processing features may not work correctly due to NLTK setup issues.")

# ===================== NetflixRecommender Setup =====================
try:
    from netflix_recommender import NetflixRecommender

    RECOMMENDER_AVAILABLE = True
    logger.info("NetflixRecommender imported successfully")
except ImportError as e:
    logger.error(f"Failed to import NetflixRecommender: {str(e)}")
    RECOMMENDER_AVAILABLE = False


    # Define fallback recommender if the main one isn't available
    class FallbackRecommender:
        def __init__(self):
            self.titles_df = None
            self.credits_df = None
            logger.info("Using FallbackRecommender")

        def log_memory_usage(self, step):
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
                logger.info(f"Memory usage at {step}: {mem:.2f} MB")
            except:
                logger.warning("Could not measure memory usage")

        def load_dataframes(self, titles_df, credits_df):
            self.titles_df = titles_df
            self.credits_df = credits_df
            self.log_memory_usage("Loaded dataframes in fallback")

        def preprocess_data(self):
            logger.warning("Using fallback recommender with limited functionality.")

        def build_models(self):
            logger.warning("Models not built fully - using fallback implementation.")

        def clear_memory(self):
            gc.collect()
            self.log_memory_usage("After fallback memory clearing")

        def get_recommendations_for_user(self, liked_ids, top_n=10, diversity_factor=0.3):
            logger.info("Using simple popularity-based recommendations (fallback mode).")
            if self.titles_df is not None:
                return [
                           {"id": row["id"], "title": row["title"], "score": row.get("combined_score", 0)}
                           for _, row in
                           self.titles_df.sort_values("combined_score", ascending=False).head(top_n * 2).iterrows()
                           if row["id"] not in liked_ids
                       ][:top_n]
            return []

        def get_content_recommendations(self, title_id, top_n=10):
            logger.warning("Content recommendations not available in fallback mode.")
            return []

        def get_cast_crew_recommendations(self, title_id, top_n=10):
            logger.warning("Cast & crew recommendations not available in fallback mode.")
            return []

        def get_popular_in_genre(self, genre, top_n=10):
            if self.titles_df is not None:
                genre_matches = []
                for _, row in self.titles_df.iterrows():
                    genres = row.get('genres', [])
                    if isinstance(genres, str):
                        try:
                            genres = json.loads(genres.replace("'", '"'))
                        except:
                            genres = [genres]
                    if genre in genres:
                        genre_matches.append({
                            "id": row["id"],
                            "title": row["title"],
                            "combined_score": row.get("combined_score", 0)
                        })
                return sorted(genre_matches, key=lambda x: x.get("combined_score", 0), reverse=True)[:top_n]
            return []


    NetflixRecommender = FallbackRecommender

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
    .metric-card {
        background-color: #E50914;
        color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
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
    .loading-spinner {
        text-align: center;
        margin: 100px auto;
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
    .modal-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
        border-bottom: 1px solid #eee;
        padding-bottom: 1rem;
    }
    .modal-title {
        color: #E50914;
        margin: 0;
    }
    .modal-section {
        margin-bottom: 1.5rem;
    }
    .modal-section-title {
        color: #221f1f;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metadata-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
    }
    .metadata-item {
        margin-bottom: 0.5rem;
    }
    .metadata-label {
        font-weight: bold;
        color: #555;
    }
    .cast-item {
        display: inline-block;
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem;
    }
    .truncate-text {
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .stStatusWidget {
        visibility: visible;
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


# ===================== Memory Monitoring =====================
def log_memory_usage(step):
    """Log memory usage at different steps"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
        logger.info(f"Memory usage at {step}: {mem:.2f} MB")
    except:
        logger.warning("Could not measure memory usage")


# ===================== Data & Model Loading =====================
GITHUB_TITLES_URL = "https://raw.githubusercontent.com/S2k22/Netflix_Recommender/master/titles.csv"
GITHUB_CREDITS_URL = "https://raw.githubusercontent.com/S2k22/Netflix_Recommender/master/credits.csv"


# Check if data needs to be reloaded due to inactivity
def check_activity_timeout():
    current_time = time.time()
    if hasattr(st.session_state, 'last_activity'):
        # If more than 15 minutes since last activity, reset recommendation engine
        if current_time - st.session_state.last_activity > 900:  # 15 minutes
            logger.info("Timeout detected, clearing memory...")
            if hasattr(st.session_state, 'recommender'):
                try:
                    st.session_state.recommender.clear_memory()
                except:
                    pass
    st.session_state.last_activity = current_time


# Update activity timestamp
def update_activity():
    st.session_state.last_activity = time.time()


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
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error when loading data: {str(e)}")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")
        return None


def load_data_and_build_models():
    try:
        if st.session_state.data_loaded and hasattr(st.session_state, 'titles_df') and hasattr(st.session_state,
                                                                                               'credits_df'):
            logger.info("Using cached data")
            titles_df = st.session_state.titles_df
            credits_df = st.session_state.credits_df
        else:
            # Load fresh data
            titles_df = load_data_from_github(GITHUB_TITLES_URL)
            credits_df = load_data_from_github(GITHUB_CREDITS_URL)
            if titles_df is None or credits_df is None:
                return False

            st.session_state.data_loaded = True
            st.session_state.titles_df = titles_df
            st.session_state.credits_df = credits_df

        log_memory_usage("After loading data")

        # Create person_ID if needed
        if 'person_ID' not in credits_df.columns:
            logger.info("Creating person_ID field from names")
            credits_df['person_ID'] = credits_df['name'].astype('category').cat.codes

        # Convert numeric columns properly
        numeric_columns = ['imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']
        for col in numeric_columns:
            if col in titles_df.columns:
                titles_df[col] = pd.to_numeric(titles_df[col], errors='coerce')
        if 'release_year' in titles_df.columns:
            titles_df['release_year'] = pd.to_numeric(titles_df['release_year'], errors='coerce')
        if 'runtime' in titles_df.columns:
            titles_df['runtime'] = pd.to_numeric(titles_df['runtime'], errors='coerce')

        # Filter to common IDs for faster processing
        common_ids = set(titles_df['id']).intersection(set(credits_df['id']))
        titles_df = titles_df[titles_df['id'].isin(common_ids)].copy()
        credits_df = credits_df[credits_df['id'].isin(common_ids)].copy()
        titles_df.reset_index(drop=True, inplace=True)
        credits_df.reset_index(drop=True, inplace=True)

        # Calculate combined score
        titles_df['combined_score'] = titles_df.apply(calculate_combined_score, axis=1)

        # Sample data if too large to avoid memory issues
        max_titles = 5000  # Limit to 5000 titles
        if len(titles_df) > max_titles:
            logger.info(f"Sampling titles from {len(titles_df)} to {max_titles}")
            # Sort by combined score before sampling to keep the best titles
            titles_df = titles_df.sort_values('combined_score', ascending=False).head(max_titles)
            # Update credits to match sampled titles
            credits_df = credits_df[credits_df['id'].isin(titles_df['id'])]

        max_credits = 50000  # Limit credits to reduce memory usage
        if len(credits_df) > max_credits:
            logger.info(f"Sampling credits from {len(credits_df)} to {max_credits}")
            credits_df = credits_df.sample(max_credits, random_state=42)

        # Create recommender instance
        recommender = NetflixRecommender()
        recommender.load_dataframes(titles_df, credits_df)

        with st.spinner("Building recommendation models..."):
            recommender.preprocess_data()
            recommender.build_models()

        st.session_state.recommender = recommender
        st.session_state.titles_df = titles_df
        st.session_state.credits_df = credits_df
        st.session_state.model_built = True
        st.session_state.loading = False
        st.session_state.error_count = 0  # Reset error count on success
        log_memory_usage("After building models")
        return True
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        st.session_state.error_count += 1
        st.session_state.loading = False
        return False


def safe_rerun():
    try:
        st.rerun()
    except:
        st.error("Could not refresh the app. Please reload the page.")
        st.stop()


# ===================== UI Function for Details =====================
def show_title_details(title_id):
    """
    Displays a modal-like section with details for the given title_id.
    This is shown in the MAIN content area (center), not the sidebar.
    """
    update_activity()  # Register user activity

    try:
        title_info = st.session_state.titles_df[st.session_state.titles_df['id'] == title_id]
        if title_info.empty:
            st.error("Title information not found.")
            return

        title_data = title_info.iloc[0]
        cast_data = st.session_state.credits_df[
            (st.session_state.credits_df['id'] == title_id) &
            (st.session_state.credits_df['role'] == 'ACTOR')
            ]
        director_data = st.session_state.credits_df[
            (st.session_state.credits_df['id'] == title_id) &
            (st.session_state.credits_df['role'] == 'DIRECTOR')
            ]

        # "Modal" container in the center
        with st.container():
            st.markdown("---")
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"# {title_data['title']}")
            with col2:
                if st.button("Close", key=f"close_modal_{title_id}"):
                    st.session_state.show_detail_modal = False
                    st.session_state.detail_title_id = None
                    safe_rerun()

            st.markdown(f"**{title_data.get('type', 'N/A')}** | **Released:** {title_data.get('release_year', 'N/A')}")
            col_desc, col_meta = st.columns([3, 2])

            with col_desc:
                st.markdown("### Description")
                description = title_data.get('description', 'No description available.')
                st.markdown(f"{description}")

                genres = title_data.get('genres', [])
                if isinstance(genres, list) and genres:
                    genre_str = ', '.join(genres)
                elif isinstance(genres, str):
                    try:
                        genres_list = json.loads(genres.replace("'", '"'))
                        genre_str = ', '.join(genres_list)
                    except:
                        genre_str = genres
                else:
                    genre_str = 'N/A'
                st.markdown("### Genres")
                st.markdown(f"{genre_str}")

                # Cast
                if not cast_data.empty:
                    st.markdown("### Cast")
                    cast_list = [actor['name'] for _, actor in cast_data.head(10).iterrows()]
                    st.markdown(', '.join(cast_list))

                # Directors
                if not director_data.empty:
                    st.markdown("### Director(s)")
                    dir_list = [director['name'] for _, director in director_data.iterrows()]
                    st.markdown(', '.join(dir_list))

            with col_meta:
                st.markdown("### Ratings & Info")
                imdb_score = title_data.get('imdb_score', 'N/A')
                if pd.notna(imdb_score):
                    st.markdown(f"**IMDB Score:** {imdb_score}")

                imdb_votes = title_data.get('imdb_votes', 'N/A')
                if pd.notna(imdb_votes):
                    st.markdown(f"**IMDB Votes:** {int(imdb_votes):,}")

                tmdb_score = title_data.get('tmdb_score', 'N/A')
                if pd.notna(tmdb_score):
                    st.markdown(f"**TMDB Score:** {tmdb_score}")

                tmdb_popularity = title_data.get('tmdb_popularity', 'N/A')
                if pd.notna(tmdb_popularity):
                    st.markdown(f"**TMDB Popularity:** {tmdb_popularity}")

                combined_score = title_data.get('combined_score', 'N/A')
                if pd.notna(combined_score):
                    st.markdown(f"**Combined Score:** {combined_score:.2f}")

                runtime = title_data.get('runtime', 'N/A')
                if pd.notna(runtime):
                    st.markdown(f"**Runtime:** {int(runtime)} min")

                st.markdown("---")
                # Let user select or deselect from here as well
                is_selected = any(item['id'] == title_id for item in st.session_state.selected_titles)
                if is_selected:
                    if st.button("Deselect", key=f"detail_deselect_{title_id}"):
                        st.session_state.selected_titles = [
                            item for item in st.session_state.selected_titles if item['id'] != title_id
                        ]
                        safe_rerun()
                else:
                    if st.button("Select", key=f"detail_select_{title_id}"):
                        st.session_state.selected_titles.append({'id': title_id, 'title': title_data['title']})
                        safe_rerun()

            st.markdown("---")
    except Exception as e:
        st.error(f"Error displaying title details: {str(e)}")
        logger.error(f"Error in show_title_details: {str(e)}")
        logger.error(traceback.format_exc())


# ===================== Recovery Mechanism =====================
def try_recover():
    """Try to recover from error state"""
    if st.session_state.error_count > 3:
        logger.warning("Multiple errors detected, attempting recovery...")
        st.warning("The app has encountered multiple errors. Attempting to recover...")

        # Clear any existing session data that might be corrupted
        if hasattr(st.session_state, 'recommender'):
            try:
                st.session_state.recommender.clear_memory()
            except:
                pass

        # Force garbage collection
        gc.collect()

        # Reset state
        st.session_state.model_built = False
        st.session_state.loading = True
        st.session_state.error_count = 0

        # Attempt to reload
        load_data_and_build_models()
        return True
    return False


# ===================== MAIN APP LAYOUT =====================
try:
    # Check if we need timeout-based recovery
    check_activity_timeout()

    # Show header
    st.markdown("<h1 class='main-header'>Netflix Advanced Recommender System</h1>", unsafe_allow_html=True)

    # If multiple errors, try recovery
    if st.session_state.error_count > 3:
        try_recover()

    # ======== SIDEBAR: Config & Data Loading ========
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/7a/Logonetflix.png", width=200)
        st.markdown("### Recommendation Settings")

        recommendation_type = st.radio(
            "Select how recommendations are generated:",
            ["Hybrid (Default)", "Content-Based", "Cast & Crew Based"],
            key="recommendation_type_radio",
            help="Hybrid combines all signals, Content-Based focuses on plot and genres, Cast & Crew focuses on actors and directors"
        )
        if recommendation_type == "Hybrid (Default)":
            st.session_state.recommendation_type = "hybrid"
        elif recommendation_type == "Content-Based":
            st.session_state.recommendation_type = "content"
        else:
            st.session_state.recommendation_type = "cast_crew"

        diversity_factor = st.slider(
            "Diversity factor", 0.0, 1.0, 0.3, 0.1,
            help="Higher values prioritize diversity over similarity"
        )
        num_recommendations = st.slider("Number of recommendations", 5, 20, 10)

        # Performance settings
        st.markdown("### Performance Settings")
        if st.button("Clear Memory"):
            try:
                if hasattr(st.session_state, 'recommender'):
                    st.session_state.recommender.clear_memory()
                gc.collect()
                st.success("Memory cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing memory: {str(e)}")

        # About section
        st.markdown("### About")
        st.markdown("""
        This recommender system analyzes Netflix content to provide personalized recommendations.

        Features:
        - Content-based filtering
        - Cast & crew similarity
        - Popularity recommendations
        - Hybrid approaches
        - Diverse recommendations

        Dataset includes 5000+ titles and 77k+ actor/director credits.
        """)

        # If still loading data, do it here
        if st.session_state.loading:
            with st.spinner(st.session_state.loading_message):
                if load_data_and_build_models():
                    st.success("Models built successfully!")
                    safe_rerun()
                else:
                    # If loading failed but we haven't tried recovery yet, try again
                    if st.session_state.error_count > 2 and st.session_state.error_count < 5:
                        st.warning("Loading failed, attempting recovery...")
                        try_recover()
        elif not st.session_state.model_built:
            st.error("Failed to load recommendation models. Please refresh the page to try again.")
            if st.button("Try Again"):
                st.session_state.loading = True
                safe_rerun()
        else:
            # The user can keep adjusting these settings at any time
            pass

    # ======== MAIN CONTENT: TABS, DETAILS, & RECOMMENDATIONS ========

    # If data not loaded, just stop
    if not st.session_state.model_built:
        st.warning("Loading recommendation models... Please wait.")
        st.stop()

    # If we have a detail modal to show, do it first so it's centered
    if st.session_state.show_detail_modal and st.session_state.detail_title_id:
        show_title_details(st.session_state.detail_title_id)

    # Are we in "recommendation mode" or "selection mode"?
    if not st.session_state.recommendation_mode:
        # ======== SELECTION MODE ========
        tab1, tab2 = st.tabs(["Top Rated Titles", "Popular by Genre"])

        with tab1:
            st.markdown("<h2 class='sub-header'>Select Titles You Like</h2>", unsafe_allow_html=True)
            st.markdown("Choose a few titles you enjoy, and we'll recommend similar content you might like.")

            if not st.session_state.titles_df.empty:
                top_titles = st.session_state.titles_df.sort_values('combined_score', ascending=False).head(20)
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
                                if isinstance(genres, list) and genres:
                                    st.markdown(f"**Genres:** {', '.join(genres[:3])}")
                                elif isinstance(genres, str):
                                    try:
                                        genres_list = json.loads(genres.replace("'", '"'))
                                        st.markdown(f"**Genres:** {', '.join(genres_list[:3])}")
                                    except:
                                        st.markdown(f"**Genres:** {genres}")

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

                                if st.button("View Details", key=f"view_{title['id']}"):
                                    update_activity()  # Register user activity
                                    st.session_state.show_detail_modal = True
                                    st.session_state.detail_title_id = title['id']
                                    safe_rerun()

                                title_id = title['id']
                                title_name = title['title']
                                is_selected = any(item['id'] == title_id for item in st.session_state.selected_titles)
                                if is_selected:
                                    if st.button("Deselect", key=f"deselect_{title_id}"):
                                        update_activity()  # Register user activity
                                        st.session_state.selected_titles = [
                                            item for item in st.session_state.selected_titles if item['id'] != title_id
                                        ]
                                        safe_rerun()
                                else:
                                    if st.button("Select", key=f"select_{title_id}"):
                                        update_activity()  # Register user activity
                                        st.session_state.selected_titles.append({
                                            'id': title_id,
                                            'title': title_name
                                        })
                                        safe_rerun()

        with tab2:
            st.markdown("<h2 class='sub-header'>Browse Popular Titles by Genre</h2>", unsafe_allow_html=True)
            st.markdown("Explore top-rated titles in specific genres and add them to your selection.")

            all_genres = []
            for g in st.session_state.titles_df['genres']:
                if isinstance(g, list):
                    all_genres.extend(g)
                elif isinstance(g, str):
                    try:
                        genres_list = json.loads(g.replace("'", '"'))
                        all_genres.extend(genres_list)
                    except:
                        pass

            unique_genres = sorted(list(set(all_genres)))
            st.markdown("### Select a Genre")

            genres_per_row = 5
            genre_rows = [unique_genres[i:i + genres_per_row] for i in range(0, len(unique_genres), genres_per_row)]
            for row in genre_rows:
                cols = st.columns(genres_per_row)
                for i, genre in enumerate(row):
                    with cols[i]:
                        if st.button(genre, key=f"genre_{genre}", help=f"Show popular titles in {genre} genre"):
                            update_activity()  # Register user activity
                            st.session_state.selected_genre = genre
                            safe_rerun()

            if st.session_state.selected_genre:
                st.markdown(f"### Popular Titles in {st.session_state.selected_genre}")
                try:
                    popular_in_genre = st.session_state.recommender.get_popular_in_genre(
                        st.session_state.selected_genre, top_n=12)

                    if popular_in_genre:
                        cols_per_row = 3
                        for i in range(0, len(popular_in_genre), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j in range(cols_per_row):
                                idx = i + j
                                if idx < len(popular_in_genre):
                                    rec = popular_in_genre[idx]
                                    title_info = st.session_state.titles_df[
                                        st.session_state.titles_df['id'] == rec['id']]
                                    if not title_info.empty:
                                        title_details = title_info.iloc[0]
                                        with cols[j]:
                                            with st.container():
                                                st.markdown(f"### {rec['title']}")
                                                st.markdown(
                                                    f"**{title_details.get('show_type', 'N/A')} "
                                                    f"({title_details.get('release_year', 'N/A')})**"
                                                )

                                                genres_val = title_details.get('genres', [])
                                                if isinstance(genres_val, list) and genres_val:
                                                    st.markdown(f"**Genres:** {', '.join(genres_val[:3])}")
                                                elif isinstance(genres_val, str):
                                                    try:
                                                        genres_list = json.loads(genres_val.replace("'", '"'))
                                                        st.markdown(f"**Genres:** {', '.join(genres_list[:3])}")
                                                    except:
                                                        st.markdown(f"**Genres:** {genres_val}")

                                                description = title_details.get('description', '')
                                                if description:
                                                    st.markdown(
                                                        f"<div class='truncate-text'>{description}</div>",
                                                        unsafe_allow_html=True
                                                    )

                                                imdb_score = title_details['imdb_score'] if pd.notna(
                                                    title_details['imdb_score']) else "N/A"
                                                combined_score = title_details['combined_score'] if pd.notna(
                                                    title_details['combined_score']) else "N/A"
                                                combined_score_formatted = f"{combined_score:.2f}" if isinstance(
                                                    combined_score, float) else combined_score
                                                st.markdown(
                                                    f"**IMDB Score:** {imdb_score} "
                                                    f"<span class='combined-score-badge'>Combined Score: {combined_score_formatted}</span>",
                                                    unsafe_allow_html=True
                                                )

                                                if st.button("View Details", key=f"view_genre_{rec['id']}"):
                                                    update_activity()  # Register user activity
                                                    st.session_state.show_detail_modal = True
                                                    st.session_state.detail_title_id = rec['id']
                                                    safe_rerun()

                                                title_id = rec['id']
                                                title_name = rec['title']
                                                is_selected = any(
                                                    item['id'] == title_id for item in st.session_state.selected_titles)
                                                if is_selected:
                                                    if st.button("Deselect", key=f"genre_deselect_{title_id}"):
                                                        update_activity()  # Register user activity
                                                        st.session_state.selected_titles = [
                                                            item for item in st.session_state.selected_titles
                                                            if item['id'] != title_id
                                                        ]
                                                        safe_rerun()
                                                else:
                                                    if st.button("Select", key=f"genre_select_{title_id}"):
                                                        update_activity()  # Register user activity
                                                        st.session_state.selected_titles.append({
                                                            'id': title_id,
                                                            'title': title_name
                                                        })
                                                        safe_rerun()
                    else:
                        st.warning(f"No titles found in the {st.session_state.selected_genre} genre.")
                except Exception as e:
                    st.error(f"Error loading genre recommendations: {str(e)}")
                    logger.error(f"Error in genre recommendations: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Attempt recovery if needed
                    st.session_state.error_count += 1
                    if try_recover():
                        safe_rerun()

        # Show the "Get Recommendations" button outside the tabs
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Your Selections")
        if st.session_state.selected_titles:
            st.markdown("You've selected:")
            for i, item in enumerate(st.session_state.selected_titles):
                st.markdown(f"{i + 1}. {item['title']}")
            # The button for final recommendations:
            if st.button("Get Recommendations", key="get_recs_button"):
                update_activity()  # Register user activity
                st.session_state.recommendation_mode = True
                safe_rerun()
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
            update_activity()  # Register user activity
            st.session_state.recommendation_mode = False
            st.session_state.selected_titles = []
            safe_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        liked_ids = [item['id'] for item in st.session_state.selected_titles]

        try:
            if st.session_state.recommendation_type == "content":
                st.markdown("<div class='recommendation-type-selector'>", unsafe_allow_html=True)
                st.markdown("🔍 **Using Content-Based recommendations** - focusing on plot, genres, and themes")
                st.markdown("</div>", unsafe_allow_html=True)
                all_recommendations = []
                for title_id in liked_ids:
                    try:
                        content_recs = st.session_state.recommender.get_content_recommendations(title_id, top_n=20)
                        all_recommendations.extend(content_recs)
                    except Exception as e:
                        logger.warning(f"Couldn't get recommendations for title {title_id}: {str(e)}")
                filtered_recs = {}
                for rec in all_recommendations:
                    if rec['id'] not in liked_ids and rec['id'] not in filtered_recs:
                        filtered_recs[rec['id']] = rec
                recommendations = sorted(filtered_recs.values(), key=lambda x: x['similarity_score'], reverse=True)[
                                  :num_recommendations]

            elif st.session_state.recommendation_type == "cast_crew":
                st.markdown("<div class='recommendation-type-selector'>", unsafe_allow_html=True)
                st.markdown("🎭 **Using Cast & Crew recommendations** - focusing on actors and directors you might like")
                st.markdown("</div>", unsafe_allow_html=True)
                all_recommendations = []
                for title_id in liked_ids:
                    try:
                        cast_crew_recs = st.session_state.recommender.get_cast_crew_recommendations(title_id, top_n=20)
                        all_recommendations.extend(cast_crew_recs)
                    except Exception as e:
                        logger.warning(f"Couldn't get recommendations for title {title_id}: {str(e)}")
                filtered_recs = {}
                for rec in all_recommendations:
                    if rec['id'] not in liked_ids and rec['id'] not in filtered_recs:
                        filtered_recs[rec['id']] = rec
                recommendations = sorted(filtered_recs.values(), key=lambda x: x['similarity_score'], reverse=True)[
                                  :num_recommendations]

            else:
                st.markdown("<div class='recommendation-type-selector'>", unsafe_allow_html=True)
                st.markdown("🔄 **Using Hybrid recommendations** - balancing content, cast & crew, and quality metrics")
                st.markdown("</div>", unsafe_allow_html=True)

                # With progress bar for better user feedback
                with st.spinner("Generating hybrid recommendations..."):
                    recommendations = st.session_state.recommender.get_recommendations_for_user(
                        liked_ids, top_n=num_recommendations, diversity_factor=diversity_factor)

            st.markdown("<h3 class='sub-header'>Recommendations For You</h3>", unsafe_allow_html=True)
            if not recommendations:
                st.warning(
                    "No recommendations found. Try selecting different titles or a different recommendation type.")
            else:
                cols_per_row = 3
                for i in range(0, len(recommendations), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < len(recommendations):
                            rec = recommendations[idx]
                            rec_info = st.session_state.titles_df[st.session_state.titles_df['id'] == rec['id']]
                            if not rec_info.empty:
                                rec_details = rec_info.iloc[0]
                                with cols[j]:
                                    with st.container():
                                        st.markdown(f"### {idx + 1}. {rec['title']}")
                                        st.markdown(
                                            f"**{rec_details.get('show_type', 'N/A')} "
                                            f"({rec_details.get('release_year', 'N/A')})**"
                                        )

                                        genres_val = rec_details.get('genres', [])
                                        if isinstance(genres_val, list) and genres_val:
                                            st.markdown(f"**Genres:** {', '.join(genres_val[:3])}")
                                        elif isinstance(genres_val, str):
                                            try:
                                                genres_list = json.loads(genres_val.replace("'", '"'))
                                                st.markdown(f"**Genres:** {', '.join(genres_list[:3])}")
                                            except:
                                                st.markdown(f"**Genres:** {genres_val}")

                                        description = rec_details.get('description', '')
                                        if description:
                                            st.markdown(f"<div class='truncate-text'>{description}</div>",
                                                        unsafe_allow_html=True)

                                        if st.session_state.recommendation_type == "hybrid":
                                            score = rec.get('score', 0)
                                            score_text = f"<span class='score-badge'>Score: {score:.2f}</span>"
                                        else:
                                            similarity = rec.get('similarity_score', 0)
                                            score_text = f"<span class='score-badge'>Similarity: {similarity:.2f}</span>"

                                        combined_score = rec_details.get('combined_score', None)
                                        if combined_score is not None and pd.notna(combined_score):
                                            combined_score_text = f"<span class='combined-score-badge'>Combined Score: {combined_score:.2f}</span>"
                                        else:
                                            combined_score_text = ""

                                        st.markdown(f"{score_text} {combined_score_text}", unsafe_allow_html=True)

                                        if st.button("View Details", key=f"view_rec_{rec['id']}"):
                                            update_activity()  # Register user activity
                                            st.session_state.show_detail_modal = True
                                            st.session_state.detail_title_id = rec['id']
                                            safe_rerun()
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            logger.error(f"Error in recommendation mode: {str(e)}")
            logger.error(traceback.format_exc())
            st.session_state.error_count += 1

            # Show error recovery options
            st.markdown("<div class='system-message'>", unsafe_allow_html=True)
            st.markdown(
                "The system encountered an error while generating recommendations. This might be due to memory constraints.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Try Again"):
                    update_activity()
                    # Force garbage collection
                    gc.collect()
                    safe_rerun()
            with col2:
                if st.button("Go Back to Selection"):
                    update_activity()
                    st.session_state.recommendation_mode = False
                    # Force garbage collection
                    gc.collect()
                    safe_rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            # Try recovery if we've had multiple errors
            if try_recover():
                safe_rerun()

    # Log memory usage occasionally
    if not hasattr(st.session_state, 'last_memory_log') or time.time() - st.session_state.last_memory_log > 60:
        log_memory_usage("Regular interval")
        st.session_state.last_memory_log = time.time()

    # Footer
    st.markdown("<div class='footer'>Netflix Advanced Recommender System - Created with Streamlit</div>",
                unsafe_allow_html=True)

except Exception as e:
    # Global error handling
    st.error(f"An unexpected error occurred: {str(e)}")
    logger.error(f"Global error: {str(e)}")
    logger.error(traceback.format_exc())
    st.session_state.error_count += 1

    st.markdown("<div class='system-message'>", unsafe_allow_html=True)
    st.markdown("### System Error")
    st.markdown(
        "The application has encountered an unexpected error. This might be due to memory issues or a temporary problem.")

    if st.button("Attempt Recovery"):
        # Reset critical state
        st.session_state.loading = True
        st.session_state.model_built = False
        if hasattr(st.session_state, 'recommender'):
            try:
                del st.session_state.recommender
            except:
                pass
        # Force garbage collection
        gc.collect()
        safe_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # If we've had too many errors, try more aggressive recovery
    if st.session_state.error_count > 5:
        try_recover()
