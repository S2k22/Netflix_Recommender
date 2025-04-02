import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.stem.snowball import SnowballStemmer
import json
import re
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('netflix_recommender')

# Download NLTK resources with error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download issue: {e}")


class NetflixRecommender:
    def __init__(self, titles_df=None, credits_df=None):
        self.titles_df = titles_df
        self.credits_df = credits_df
        self.preprocessed = False
        self.model_built = False

        # Initialize components
        self.content_similarity = None
        self.actor_director_similarity = None
        self.popularity_scores = None
        self.combined_similarity = None

        # Track memory usage
        self.log_memory_usage("Initialized recommender")

    def log_memory_usage(self, step):
        """Log memory usage at different steps"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
            logger.info(f"Memory usage at {step}: {mem:.2f} MB")
        except:
            logger.warning("Could not measure memory usage")

    def load_dataframes(self, titles_df, credits_df):
        """Load data from existing DataFrames"""
        # Sample data if it's too large
        if len(titles_df) > 5000:
            logger.info(f"Sampling titles from {len(titles_df)} to 5000 records")
            titles_df = titles_df.sample(5000, random_state=42)

        if len(credits_df) > 50000:
            logger.info(f"Sampling credits from {len(credits_df)} to 50000 records")
            credits_df = credits_df.sample(50000, random_state=42)

        self.titles_df = titles_df
        self.credits_df = credits_df
        self.preprocessed = False
        self.model_built = False
        self.log_memory_usage("Loaded dataframes")

    def load_from_files(self, titles_path, credits_path):
        """Load data from CSV files"""
        try:
            self.titles_df = pd.read_csv(titles_path)
            self.credits_df = pd.read_csv(credits_path)
            self.preprocessed = False
            self.model_built = False
            self.log_memory_usage("Loaded data from files")
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            raise

    def clear_memory(self):
        """Clear large matrices to free up memory"""
        logger.info("Clearing memory...")
        # Save small necessary data before clearing
        self.title_idx_map_backup = self.title_idx_map if hasattr(self, 'title_idx_map') else None
        self.idx_title_map_backup = self.idx_title_map if hasattr(self, 'idx_title_map') else None
        self.id_to_title_backup = self.id_to_title if hasattr(self, 'id_to_title') else None

        # Clear large matrices
        self.content_similarity = None
        self.actor_director_similarity = None

        # Force garbage collection
        gc.collect()
        self.log_memory_usage("After memory clearing")

    def _parse_list_strings(self, df, column):
        """Parse string representations of lists to actual Python lists"""

        def parse_list(x):
            if pd.isna(x) or x == '':
                return []
            if isinstance(x, list):
                return x
            try:
                # Try to parse as JSON first
                return json.loads(x.replace("'", '"'))
            except:
                # If that fails, try a simple split
                return [item.strip() for item in x.split(',')]

        df[column] = df[column].apply(parse_list)
        return df

    def preprocess_data(self):
        """Preprocess the loaded data for recommendation models"""
        logger.info("Preprocessing data...")
        self.log_memory_usage("Before preprocessing")

        if self.titles_df is None or self.credits_df is None:
            raise ValueError("No data loaded. Please load data first using load_dataframes() or load_from_files().")

        # Create person_ID if not exists
        if 'person_ID' not in self.credits_df.columns:
            logger.info("Creating person_ID field from names")
            self.credits_df['person_ID'] = self.credits_df['name'].astype('category').cat.codes

        # Handle missing values efficiently
        self.titles_df = self.titles_df.fillna('')
        self.credits_df = self.credits_df.fillna('')

        # Add show_type if missing
        if 'show_type' not in self.titles_df.columns:
            self.titles_df['show_type'] = 'N/A'

        # Convert string lists to actual lists
        list_columns = ['genres', 'production_countries']
        for col in list_columns:
            if col in self.titles_df.columns:
                self.titles_df = self._parse_list_strings(self.titles_df, col)

        # Create mappings for faster lookups
        self.title_idx_map = {title_id: idx for idx, title_id in enumerate(self.titles_df['id'].unique())}
        self.idx_title_map = {idx: title_id for title_id, idx in self.title_idx_map.items()}
        self.id_to_title = dict(zip(self.titles_df['id'], self.titles_df['title']))

        # Process text efficiently
        logger.info("Processing text features...")
        stemmer = SnowballStemmer('english')

        def clean_text(text):
            if isinstance(text, str):
                # Simple cleaning for efficiency
                text = re.sub('[^a-zA-Z]', ' ', text.lower())
                words = [stemmer.stem(word) for word in nltk.word_tokenize(text) if len(word) > 2]
                return ' '.join(words)
            return ''

        # Clean description text
        self.titles_df['clean_description'] = self.titles_df['description'].apply(clean_text)

        # Convert list features to strings
        self.titles_df['genres_str'] = self.titles_df['genres'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else '')
        self.titles_df['countries_str'] = self.titles_df['production_countries'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else '')

        # Process cast and crew more efficiently
        logger.info("Processing cast and crew data...")
        # Group credits by title and role
        self.title_actors = self.credits_df[self.credits_df['role'] == 'ACTOR'].groupby('id')['name'].apply(
            list).to_dict()
        self.title_directors = self.credits_df[self.credits_df['role'] == 'DIRECTOR'].groupby('id')['name'].apply(
            list).to_dict()

        # Add actor and director strings to titles dataframe
        self.titles_df['actors_str'] = self.titles_df['id'].apply(
            lambda x: ' '.join(self.title_actors.get(x, [])) if x in self.title_actors else '')
        self.titles_df['directors_str'] = self.titles_df['id'].apply(
            lambda x: ' '.join(self.title_directors.get(x, [])) if x in self.title_directors else '')

        # Combine features efficiently
        self.titles_df['combined_features'] = (
                self.titles_df['clean_description'] + ' ' +
                self.titles_df['genres_str'] + ' ' +
                self.titles_df['countries_str'] + ' ' +
                self.titles_df['actors_str'] + ' ' +
                self.titles_df['directors_str']
        )

        self.preprocessed = True
        self.log_memory_usage("After preprocessing")
        logger.info("Preprocessing complete.")

    def calculate_combined_score(self, row):
        """Calculate a combined score from various rating metrics"""
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

    def build_models(self):
        """Build recommendation models with memory optimization"""
        if not self.preprocessed:
            self.preprocess_data()

        logger.info("Building recommendation models...")
        self.log_memory_usage("Before building models")

        # Handle numeric columns
        numeric_columns = ['imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']
        for col in numeric_columns:
            if col in self.titles_df.columns:
                self.titles_df[col] = pd.to_numeric(self.titles_df[col], errors='coerce')

        # Calculate combined score if needed
        if 'combined_score' not in self.titles_df.columns:
            logger.info("Calculating combined scores...")
            self.titles_df['combined_score'] = self.titles_df.apply(self.calculate_combined_score, axis=1)

        # 1. Content-based similarity - build in chunks for memory efficiency
        logger.info("Building content-based model...")
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.titles_df['combined_features'])

        # Build similarity matrix in chunks
        n = len(self.titles_df)
        chunk_size = min(1000, n)  # Adjust chunk size based on your memory constraints
        self.content_similarity = np.zeros((n, n))

        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunk = tfidf_matrix[i:end]
            self.content_similarity[i:end] = cosine_similarity(chunk, tfidf_matrix)
            # Force garbage collection after each chunk
            if i % (chunk_size * 3) == 0:
                gc.collect()

        self.log_memory_usage("After content similarity")

        # 2. Cast & Crew similarity with memory optimization
        logger.info("Building cast & crew model...")
        # Create actor-title matrix
        actor_title_df = self.credits_df[self.credits_df['role'] == 'ACTOR'].groupby(
            ['person_ID', 'id']).size().reset_index(name='count')

        if len(actor_title_df) > 0:
            actor_title_matrix = csr_matrix(
                (actor_title_df['count'],
                 (actor_title_df['person_ID'].astype('category').cat.codes,
                  [self.title_idx_map.get(tid, 0) for tid in actor_title_df['id']]))
            )
            # Build actor similarity with chunking
            n = actor_title_matrix.shape[1]
            chunk_size = min(1000, n)
            actor_similarity = np.zeros((n, n))

            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                chunk = actor_title_matrix.T[i:end]
                actor_similarity[i:end] = cosine_similarity(chunk, actor_title_matrix.T)
                if i % (chunk_size * 3) == 0:
                    gc.collect()
        else:
            actor_similarity = np.zeros((len(self.titles_df), len(self.titles_df)))

        self.log_memory_usage("After actor similarity")

        # Create director-title matrix
        director_title_df = self.credits_df[self.credits_df['role'] == 'DIRECTOR'].groupby(
            ['person_ID', 'id']).size().reset_index(name='count')

        if len(director_title_df) > 0:
            director_title_matrix = csr_matrix(
                (director_title_df['count'],
                 (director_title_df['person_ID'].astype('category').cat.codes,
                  [self.title_idx_map.get(tid, 0) for tid in director_title_df['id']]))
            )

            # Build director similarity with chunking
            n = director_title_matrix.shape[1]
            chunk_size = min(1000, n)
            director_similarity = np.zeros((n, n))

            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                chunk = director_title_matrix.T[i:end]
                director_similarity[i:end] = cosine_similarity(chunk, director_title_matrix.T)
                if i % (chunk_size * 3) == 0:
                    gc.collect()

            self.actor_director_similarity = 0.7 * actor_similarity + 0.3 * director_similarity
        else:
            self.actor_director_similarity = actor_similarity

        self.log_memory_usage("After director similarity")

        # 3. Popularity model
        logger.info("Building popularity model...")
        scaler = MinMaxScaler()
        imdb_scores = self.titles_df['imdb_score'].fillna(0)
        imdb_votes = self.titles_df['imdb_votes'].fillna(0)
        tmdb_pop = self.titles_df['tmdb_popularity'].fillna(0)
        tmdb_scores = self.titles_df['tmdb_score'].fillna(0)

        imdb_scores_scaled = scaler.fit_transform(imdb_scores.values.reshape(-1, 1)).flatten()
        imdb_votes_scaled = scaler.fit_transform(imdb_votes.values.reshape(-1, 1)).flatten()
        tmdb_pop_scaled = scaler.fit_transform(tmdb_pop.values.reshape(-1, 1)).flatten()
        tmdb_scores_scaled = scaler.fit_transform(tmdb_scores.values.reshape(-1, 1)).flatten()

        self.popularity_scores = (
                0.4 * imdb_scores_scaled +
                0.3 * imdb_votes_scaled +
                0.2 * tmdb_scores_scaled +
                0.1 * tmdb_pop_scaled
        )

        # 4. Combine all models into one
        logger.info("Combining models...")
        self.combined_similarity = 0.6 * self.content_similarity + 0.4 * self.actor_director_similarity

        self.model_built = True
        self.log_memory_usage("After building all models")
        logger.info("All models built successfully.")

    def _get_recommendations_by_similarity(self, title_id, similarity_matrix, top_n=10):
        """Get recommendations based on similarity matrix"""
        idx = self.title_idx_map.get(title_id)
        if idx is None:
            logger.warning(f"Title ID {title_id} not found in the dataset.")
            return []

        # Get the similarity scores for this title
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

        # Convert indices to title IDs
        title_indices = [i[0] for i in sim_scores]
        title_ids = [self.idx_title_map[idx] for idx in title_indices]

        # Create recommendation objects
        recommendations = [
            {
                'id': title_id,
                'title': self.id_to_title.get(title_id),
                'similarity_score': score
            } for title_id, (_, score) in zip(title_ids, sim_scores)
        ]
        return recommendations

    def get_content_recommendations(self, title_id, top_n=10):
        """Get content-based recommendations for a title"""
        if not self.model_built:
            self.build_models()
        return self._get_recommendations_by_similarity(title_id, self.content_similarity, top_n)

    def get_cast_crew_recommendations(self, title_id, top_n=10):
        """Get cast & crew based recommendations for a title"""
        if not self.model_built:
            self.build_models()
        return self._get_recommendations_by_similarity(title_id, self.actor_director_similarity, top_n)

    def get_hybrid_recommendations(self, title_id, top_n=10):
        """Get hybrid recommendations for a title"""
        if not self.model_built:
            self.build_models()
        return self._get_recommendations_by_similarity(title_id, self.combined_similarity, top_n)

    def get_popular_in_genre(self, genre, top_n=10):
        """Get popular titles in a specific genre"""
        if not self.model_built:
            self.build_models()

        # Find titles in the requested genre
        genre_titles = []
        for idx, row in self.titles_df.iterrows():
            genres = row['genres']
            if isinstance(genres, list) and genre in genres:
                genre_titles.append(idx)
            elif isinstance(genres, str):
                try:
                    genres_list = json.loads(genres.replace("'", '"'))
                    if genre in genres_list:
                        genre_titles.append(idx)
                except:
                    if genre in genres:
                        genre_titles.append(idx)

        if not genre_titles:
            logger.warning(f"No titles found in genre: {genre}")
            return []

        # Get popularity scores and sort
        scores = [(idx, self.popularity_scores[idx]) for idx in genre_titles]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

        # Create recommendation objects
        recommendations = [
            {
                'id': self.titles_df.iloc[idx]['id'],
                'title': self.titles_df.iloc[idx]['title'],
                'popularity_score': score
            } for idx, score in scores
        ]
        return recommendations

    def get_recommendations_for_user(self, liked_title_ids, top_n=20, diversity_factor=0.3):
        """Get personalized recommendations based on multiple liked titles"""
        if not self.model_built:
            self.build_models()

        # Get base recommendation scores from content and cast/crew similarity
        rec_scores = np.zeros(len(self.titles_df))
        for title_id in liked_title_ids:
            if title_id in self.title_idx_map:
                idx = self.title_idx_map[title_id]
                rec_scores += self.combined_similarity[idx]

        # Incorporate popularity/quality scores
        if 'combined_score' in self.titles_df.columns:
            # Get normalized combined scores
            combined_scores = self.titles_df['combined_score'].fillna(0).values
            max_score = max(combined_scores) if max(combined_scores) > 0 else 1
            normalized_combined_scores = combined_scores / max_score

            # Add influence from combined scores (weighted by diversity factor)
            rec_scores = (1 - diversity_factor) * rec_scores + diversity_factor * normalized_combined_scores

        # Convert to list of (index, score) pairs
        scores = list(enumerate(rec_scores))

        # Filter out titles the user has already liked
        liked_indices = [self.title_idx_map[tid] for tid in liked_title_ids if tid in self.title_idx_map]
        scores = [s for s in scores if s[0] not in liked_indices]

        # Sort by score and take the top N
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

        # Create recommendation objects
        recommendations = [
            {
                'id': self.idx_title_map[idx],
                'title': self.id_to_title.get(self.idx_title_map[idx]),
                'score': score
            } for idx, score in scores
        ]
        return recommendations