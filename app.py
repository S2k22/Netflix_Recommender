import streamlit as st
import pandas as pd
import json
import os
import requests
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Netflix Recommender System",
    page_icon="🎬",
    layout="wide"
)

try:
    import nltk

    # Create a directory for NLTK data and set the path
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)

    # Add the path to NLTK's data path
    nltk.data.path.append(nltk_data_dir)

    st.info("Downloading required NLTK resources...")
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('punkt_tab', download_dir=nltk_data_dir)  # <-- New addition to satisfy the lookup

    st.info(f"NLTK data path: {nltk.data.path}")
    st.info(
        f"Available NLTK data: {os.listdir(nltk_data_dir) if os.path.exists(nltk_data_dir) else 'No data directory found'}")

    # Verification step
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    st.success("NLTK resources loaded successfully!")

except LookupError as e:
    st.warning(f"First NLTK download attempt incomplete: {str(e)}")
    st.info("Trying alternative download method...")

    try:
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=False)
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=False)
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        st.success("NLTK resources loaded successfully on second attempt!")
    except Exception as inner_e:
        st.error(f"Failed to load NLTK resources: {str(inner_e)}")
        st.info("The app will continue, but text processing features may not work correctly.")

except Exception as e:
    st.error(f"Error with NLTK setup: {str(e)}")
    st.info("The app will continue, but some text processing features may not work correctly.")

except ImportError:
    st.error("Failed to import NLTK. Some features might not work properly.")
    nltk = None

# Import the netflix_recommender with error handling
try:
    from netflix_recommender import NetflixRecommender

    RECOMMENDER_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import NetflixRecommender: {str(e)}")
    RECOMMENDER_AVAILABLE = False


    # Create a simple fallback recommender class if the real one isn't available
    class FallbackRecommender:
        def __init__(self):
            self.titles_df = None
            self.credits_df = None

        def load_dataframes(self, titles_df, credits_df):
            self.titles_df = titles_df
            self.credits_df = credits_df

        def preprocess_data(self):
            st.warning("Using fallback recommender with limited functionality.")

        def build_models(self):
            st.warning("Models not built fully - using fallback implementation.")

        def get_recommendations_for_user(self, liked_ids, top_n=10, diversity_factor=0.3):
            st.warning("Using simple popularity-based recommendations (fallback mode).")
            if self.titles_df is not None:
                return [
                           {"id": row["id"], "title": row["title"], "score": row["combined_score"]}
                           for _, row in
                           self.titles_df.sort_values("combined_score", ascending=False).head(top_n * 2).iterrows()
                           if row["id"] not in liked_ids
                       ][:top_n]
            return []

        def get_content_recommendations(self, title_id, top_n=10):
            st.warning("Content recommendations not available in fallback mode.")
            return []

        def get_cast_crew_recommendations(self, title_id, top_n=10):
            st.warning("Cast & crew recommendations not available in fallback mode.")
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

# Add custom CSS
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
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing the recommender object
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
    st.session_state.titles_df = None
    st.session_state.credits_df = None
    st.session_state.model_built = False
    st.session_state.user_history = []
    st.session_state.current_recommendations = []
    st.session_state.loading = True
    st.session_state.loading_message = "Loading dataset and building recommendation models..."
    st.session_state.selected_genre = None
    st.session_state.recommendation_type = "hybrid"
    st.session_state.show_detail_modal = False
    st.session_state.detail_title_id = None

# Configure GitHub repository URLs - FIXED to use raw URLs
GITHUB_TITLES_URL = "https://raw.githubusercontent.com/S2k22/Netflix_Recommender/master/titles.csv"
GITHUB_CREDITS_URL = "https://raw.githubusercontent.com/S2k22/Netflix_Recommender/master/credits.csv"


# Calculate combined score for sorting titles
def calculate_combined_score(row):
    """Calculate a combined score using IMDB and TMDB metrics"""
    # Ensure all values are numeric and handle missing values
    imdb_score = row['imdb_score'] if pd.notna(row['imdb_score']) else 0
    imdb_votes = row['imdb_votes'] if pd.notna(row['imdb_votes']) else 0
    tmdb_popularity = row['tmdb_popularity'] if pd.notna(row['tmdb_popularity']) else 0
    tmdb_score = row['tmdb_score'] if pd.notna(row['tmdb_score']) else 0

    # Create weighted combined score
    combined_score = (
            0.4 * imdb_score +  # IMDB score has highest weight
            0.3 * (imdb_votes / 100000) +  # Normalize vote count
            0.2 * tmdb_score +  # TMDB score
            0.1 * (tmdb_popularity / 100)  # Normalize popularity
    )

    return combined_score


def load_data_from_github(url):
    """Load CSV data from GitHub URL with improved error handling"""
    try:
        st.info(f"Attempting to load data from: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Check if response contains CSV data by examining first few characters
        data_preview = response.text[:100]
        st.success(f"Data loaded successfully. Preview: {data_preview}...")

        return pd.read_csv(StringIO(response.text))
    except requests.exceptions.RequestException as e:
        st.error(f"Network error when loading data: {str(e)}")
        return None
    except pd.errors.ParserError as e:
        st.error(f"CSV parsing error: {str(e)}")
        st.info("Please check that the URL points to a valid CSV file.")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading data: {str(e)}")
        return None


def load_data_and_build_models():
    try:
        # Load the datasets from GitHub
        titles_df = load_data_from_github(GITHUB_TITLES_URL)
        credits_df = load_data_from_github(GITHUB_CREDITS_URL)

        if titles_df is None or credits_df is None:
            return False

        # Handle missing person_ID
        if 'person_ID' not in credits_df.columns:
            print("Creating person_ID field from names")
            credits_df['person_ID'] = credits_df['name'].astype('category').cat.codes

        # Fix numeric columns that might contain non-numeric values
        numeric_columns = ['imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']
        for col in numeric_columns:
            if col in titles_df.columns:
                # Convert to numeric, forcing errors to become NaN
                titles_df[col] = pd.to_numeric(titles_df[col], errors='coerce')

        # Same for release_year and runtime
        if 'release_year' in titles_df.columns:
            titles_df['release_year'] = pd.to_numeric(titles_df['release_year'], errors='coerce')

        if 'runtime' in titles_df.columns:
            titles_df['runtime'] = pd.to_numeric(titles_df['runtime'], errors='coerce')

        # Filter DataFrames for common IDs
        common_ids = set(titles_df['id']).intersection(set(credits_df['id']))
        titles_df = titles_df[titles_df['id'].isin(common_ids)].copy()
        credits_df = credits_df[credits_df['id'].isin(common_ids)].copy()
        titles_df.reset_index(drop=True, inplace=True)
        credits_df.reset_index(drop=True, inplace=True)

        # Calculate combined score for each title
        titles_df['combined_score'] = titles_df.apply(calculate_combined_score, axis=1)

        # Initialize recommender
        recommender = NetflixRecommender()
        recommender.load_dataframes(titles_df, credits_df)

        # Build models
        with st.spinner("Building recommendation models..."):
            recommender.preprocess_data()
            recommender.build_models()

        # Store in session state
        st.session_state.recommender = recommender
        st.session_state.titles_df = titles_df
        st.session_state.credits_df = credits_df
        st.session_state.model_built = True
        st.session_state.loading = False

        return True
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.session_state.loading = False
        return False


def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.error("Your version of Streamlit does not support experimental_rerun. Please upgrade Streamlit.")
        st.stop()


# Function to display title details modal
def show_title_details(title_id):
    """
    Displays a modal-like section with details for the given title_id.
    (Fixed to use title_id instead of rec['id'].)
    """
    # Get title info
    title_info = st.session_state.titles_df[st.session_state.titles_df['id'] == title_id]
    if title_info.empty:
        st.error("Title information not found.")
        return

    title_data = title_info.iloc[0]

    # Get cast and crew
    cast_data = st.session_state.credits_df[
        (st.session_state.credits_df['id'] == title_id) &
        (st.session_state.credits_df['role'] == 'ACTOR')
        ]

    director_data = st.session_state.credits_df[
        (st.session_state.credits_df['id'] == title_id) &
        (st.session_state.credits_df['role'] == 'DIRECTOR')
        ]

    # Create a container for the modal
    with st.container():
        st.markdown("---")
        # Create columns for layout
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"# {title_data['title']}")
        with col2:
            if st.button("Close", key=f"close_modal_{title_id}"):
                st.session_state.show_detail_modal = False
                st.session_state.detail_title_id = None
                st.rerun()

        st.markdown(f"**{title_data.get('show_type', 'N/A')}** | **Released:** {title_data.get('release_year', 'N/A')}")

        # Content grid: Description on left, metadata on right
        col_desc, col_meta = st.columns([3, 2])

        with col_desc:
            # Description
            st.markdown("### Description")
            description = title_data.get('description', 'No description available.')
            st.markdown(f"{description}")

            # Genres
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
                cast_list = []
                for _, actor in cast_data.head(10).iterrows():
                    cast_list.append(actor['name'])
                st.markdown(', '.join(cast_list))

            # Directors
            if not director_data.empty:
                st.markdown("### Director(s)")
                dir_list = []
                for _, director in director_data.iterrows():
                    dir_list.append(director['name'])
                st.markdown(', '.join(dir_list))

        with col_meta:
            # Scores and metadata
            st.markdown("### Ratings & Info")

            # IMDB Score
            imdb_score = title_data.get('imdb_score', 'N/A')
            if pd.notna(imdb_score):
                st.markdown(f"**IMDB Score:** {imdb_score}")

            # IMDB Votes
            imdb_votes = title_data.get('imdb_votes', 'N/A')
            if pd.notna(imdb_votes):
                st.markdown(f"**IMDB Votes:** {int(imdb_votes):,}")

            # TMDB Score
            tmdb_score = title_data.get('tmdb_score', 'N/A')
            if pd.notna(tmdb_score):
                st.markdown(f"**TMDB Score:** {tmdb_score}")

            # TMDB Popularity
            tmdb_popularity = title_data.get('tmdb_popularity', 'N/A')
            if pd.notna(tmdb_popularity):
                st.markdown(f"**TMDB Popularity:** {tmdb_popularity}")

            # Combined Score
            combined_score = title_data.get('combined_score', 'N/A')
            if pd.notna(combined_score):
                st.markdown(f"**Combined Score:** {combined_score:.2f}")

            # Runtime
            runtime = title_data.get('runtime', 'N/A')
            if pd.notna(runtime):
                st.markdown(f"**Runtime:** {int(runtime)} min")

            # Selection button
            st.markdown("---")
            is_selected = any(item['id'] == title_id for item in st.session_state.selected_titles)
            if is_selected:
                if st.button("Deselect", key=f"detail_deselect_{title_id}"):
                    st.session_state.selected_titles = [
                        item for item in st.session_state.selected_titles if item['id'] != title_id
                    ]
                    st.rerun()
            else:
                if st.button("Select", key=f"detail_select_{title_id}"):
                    title_name = title_data['title']
                    st.session_state.selected_titles.append({
                        'id': title_id,
                        'title': title_name
                    })
                    st.rerun()
        st.markdown("---")


# Header
st.markdown("<h1 class='main-header'>Netflix Advanced Recommender System</h1>", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/7a/Logonetflix.png", width=200)

    st.markdown("### Recommendation Settings")

    # Recommendation Type Selector
    st.markdown("#### Recommendation Type")
    recommendation_type = st.radio(
        "Select how recommendations are generated:",
        ["Hybrid (Default)", "Content-Based", "Cast & Crew Based"],
        key="recommendation_type_radio",
        help="Hybrid combines all signals, Content-Based focuses on plot and genres, Cast & Crew focuses on actors and directors"
    )

    # Map the radio selection to internal values
    if recommendation_type == "Hybrid (Default)":
        st.session_state.recommendation_type = "hybrid"
    elif recommendation_type == "Content-Based":
        st.session_state.recommendation_type = "content"
    else:
        st.session_state.recommendation_type = "cast_crew"

    # Existing diversity factor
    diversity_factor = st.slider("Diversity factor", 0.0, 1.0, 0.3, 0.1,
                                 help="Higher values prioritize diversity over similarity")

    num_recommendations = st.slider("Number of recommendations", 5, 20, 10)

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

    if st.session_state.loading:
        with st.spinner(st.session_state.loading_message):
            if load_data_and_build_models():
                st.success("Models built successfully!")
                st.rerun()  # Using st.rerun() here
    elif not st.session_state.model_built:
        st.error("Failed to load recommendation models. Please refresh the page to try again.")
    else:
        if 'recommendation_mode' not in st.session_state:
            st.session_state.recommendation_mode = False
            st.session_state.selected_titles = []

        # Check if detail modal should be shown
        if st.session_state.show_detail_modal and st.session_state.detail_title_id:
            show_title_details(st.session_state.detail_title_id)

        # INITIAL SELECTION MODE
        if not st.session_state.recommendation_mode:
            # Add tabs for different selection methods
            tab1, tab2 = st.tabs(["Top Rated Titles", "Popular by Genre"])

            with tab1:
                st.markdown("<h2 class='sub-header'>Select Titles You Like</h2>", unsafe_allow_html=True)
                st.markdown("Choose a few titles you enjoy, and we'll recommend similar content you might like.")

                # Get top-rated titles
                if not st.session_state.titles_df.empty:
                    top_titles = st.session_state.titles_df.sort_values('combined_score', ascending=False).head(20)
                else:
                    st.error("No titles available to display!")
                    top_titles = pd.DataFrame()

                # Use a multi-column layout
                cols_per_row = 3
                for i in range(0, len(top_titles), cols_per_row):
                    cols = st.columns(cols_per_row)

                    # Process titles for this row
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < len(top_titles):
                            title = top_titles.iloc[idx]
                            with cols[j]:
                                # Create card
                                with st.container():
                                    st.markdown(f"### {title['title']}")
                                    st.markdown(
                                        f"**{title.get('show_type', 'N/A')} ({title.get('release_year', 'N/A')})**")

                                    # Add genres
                                    genres = title['genres']
                                    if isinstance(genres, list) and genres:
                                        st.markdown(f"**Genres:** {', '.join(genres[:3])}")
                                    elif isinstance(genres, str):
                                        try:
                                            genres_list = json.loads(genres.replace("'", '"'))
                                            st.markdown(f"**Genres:** {', '.join(genres_list[:3])}")
                                        except Exception:
                                            st.markdown(f"**Genres:** {genres}")

                                    # Add description (truncated)
                                    description = title.get('description', '')
                                    if description:
                                        st.markdown(f"<div class='truncate-text'>{description}</div>",
                                                    unsafe_allow_html=True)

                                    # Add scores
                                    imdb_score = title['imdb_score'] if pd.notna(title['imdb_score']) else "N/A"
                                    combined_score = title['combined_score'] if pd.notna(
                                        title['combined_score']) else "N/A"
                                    combined_score_formatted = f"{combined_score:.2f}" if isinstance(combined_score,
                                                                                                     float) else combined_score

                                    st.markdown(
                                        f"**IMDB Score:** {imdb_score} "
                                        f"<span class='combined-score-badge'>Combined Score: {combined_score_formatted}</span>",
                                        unsafe_allow_html=True
                                    )

                                    # Add view details button
                                    if st.button("View Details", key=f"view_{title['id']}"):
                                        st.session_state.show_detail_modal = True
                                        st.session_state.detail_title_id = title['id']
                                        st.rerun()  # Updated rerun call

                                    # Add select/deselect button
                                    title_id = title['id']
                                    title_name = title['title']
                                    is_selected = any(
                                        item['id'] == title_id for item in st.session_state.selected_titles)
                                    if is_selected:
                                        if st.button("Deselect", key=f"deselect_{title_id}"):
                                            st.session_state.selected_titles = [
                                                item for item in st.session_state.selected_titles if
                                                item['id'] != title_id
                                            ]
                                            st.rerun()  # Updated rerun call
                                    else:
                                        if st.button("Select", key=f"select_{title_id}"):
                                            st.session_state.selected_titles.append({
                                                'id': title_id,
                                                'title': title_name
                                            })
                                            st.rerun()

            # Popular by Genre tab
            with tab2:
                st.markdown("<h2 class='sub-header'>Browse Popular Titles by Genre</h2>", unsafe_allow_html=True)
                st.markdown("Explore top-rated titles in specific genres and add them to your selection.")

                # Extract unique genres
                all_genres = []
                for genres in st.session_state.titles_df['genres']:
                    if isinstance(genres, list):
                        all_genres.extend(genres)
                    elif isinstance(genres, str):
                        try:
                            genres_list = json.loads(genres.replace("'", '"'))
                            all_genres.extend(genres_list)
                        except:
                            pass

                unique_genres = sorted(list(set(all_genres)))

                # Display genre selection pills
                st.markdown("### Select a Genre")

                # Create rows of genre pills (5 per row)
                genres_per_row = 5
                genre_rows = [unique_genres[i:i + genres_per_row] for i in range(0, len(unique_genres), genres_per_row)]

                for row in genre_rows:
                    cols = st.columns(genres_per_row)
                    for i, genre in enumerate(row):
                        with cols[i]:
                            if st.button(genre, key=f"genre_{genre}",
                                         help=f"Show popular titles in {genre} genre"):
                                st.session_state.selected_genre = genre
                                st.rerun()  # Updated rerun call

                # Display popular titles in selected genre
                if st.session_state.selected_genre:
                    st.markdown(f"### Popular Titles in {st.session_state.selected_genre}")

                    # Get popular titles in selected genre
                    popular_in_genre = st.session_state.recommender.get_popular_in_genre(
                        st.session_state.selected_genre, top_n=12)

                    if popular_in_genre:
                        # Display in grid layout
                        cols_per_row = 3
                        for i in range(0, len(popular_in_genre), cols_per_row):
                            cols = st.columns(cols_per_row)

                            # Process titles for this row
                            for j in range(cols_per_row):
                                idx = i + j
                                if idx < len(popular_in_genre):
                                    rec = popular_in_genre[idx]
                                    title_info = st.session_state.titles_df[
                                        st.session_state.titles_df['id'] == rec['id']]
                                    if not title_info.empty:
                                        title_details = title_info.iloc[0]

                                        with cols[j]:
                                            # Create card
                                            with st.container():
                                                st.markdown(f"### {rec['title']}")
                                                st.markdown(
                                                    f"**{title_details.get('show_type', 'N/A')} ({title_details.get('release_year', 'N/A')})**"
                                                )

                                                # Add genres
                                                genres = title_details.get('genres', [])
                                                if isinstance(genres, list) and genres:
                                                    st.markdown(f"**Genres:** {', '.join(genres[:3])}")
                                                elif isinstance(genres, str):
                                                    try:
                                                        genres_list = json.loads(genres.replace("'", '"'))
                                                        st.markdown(f"**Genres:** {', '.join(genres_list[:3])}")
                                                    except Exception:
                                                        st.markdown(f"**Genres:** {genres}")

                                                # Add description (truncated)
                                                description = title_details.get('description', '')
                                                if description:
                                                    st.markdown(
                                                        f"<div class='truncate-text'>{description}</div>",
                                                        unsafe_allow_html=True
                                                    )

                                                # Add scores
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

                                                # Add view details button
                                                if st.button("View Details", key=f"view_genre_{rec['id']}"):
                                                    st.session_state.show_detail_modal = True
                                                    st.session_state.detail_title_id = rec['id']
                                                    st.rerun()  # Updated rerun call

                                                # Add select/deselect button
                                                title_id = rec['id']
                                                title_name = rec['title']
                                                is_selected = any(
                                                    item['id'] == title_id for item in st.session_state.selected_titles
                                                )
                                                if is_selected:
                                                    if st.button("Deselect", key=f"genre_deselect_{title_id}"):
                                                        st.session_state.selected_titles = [
                                                            item for item in st.session_state.selected_titles
                                                            if item['id'] != title_id
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

        # Selection summary and recommendation button
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Your Selections")
        if st.session_state.selected_titles:
            st.markdown("You've selected:")
            for i, item in enumerate(st.session_state.selected_titles):
                st.markdown(f"{i + 1}. {item['title']}")
            if st.button("Get Recommendations", key="get_recs_button"):
                st.session_state.recommendation_mode = True
                st.rerun()
        else:
            st.markdown("You haven't selected any titles yet. Please select at least one title to get recommendations.")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>Netflix Advanced Recommender System - Created with Streamlit</div>",
            unsafe_allow_html=True)
