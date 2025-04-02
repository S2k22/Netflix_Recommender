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
import ast

nltk.download('punkt')
nltk.download('stopwords')


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

    def load_dataframes(self, titles_df, credits_df):

        self.titles_df = titles_df
        self.credits_df = credits_df
        self.preprocessed = False
        self.model_built = False

    def load_from_files(self, titles_path, credits_path):

        self.titles_df = pd.read_csv(titles_path)
        self.credits_df = pd.read_csv(credits_path)
        self.preprocessed = False
        self.model_built = False

    def _parse_list_strings(self, df, column):

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
        print("Preprocessing data...")

        if self.titles_df is None or self.credits_df is None:
            raise ValueError("No data loaded. Please load data first using load_dataframes() or load_from_files().")

        if 'person_ID' not in self.credits_df.columns:
            print("Creating person_ID field from names")
            self.credits_df['person_ID'] = self.credits_df['name'].astype('category').cat.codes

        # Handle missing values
        self.titles_df = self.titles_df.fillna('')
        self.credits_df = self.credits_df.fillna('')

        # If the 'show_type' column is missing, add it with a default value
        if 'show_type' not in self.titles_df.columns:
            self.titles_df['show_type'] = 'N/A'

        # Convert string representations of lists to actual lists
        list_columns = ['genres', 'production_countries']
        for col in list_columns:
            if col in self.titles_df.columns:
                self.titles_df = self._parse_list_strings(self.titles_df, col)

        # Create title ID mapping for faster lookups
        self.title_idx_map = {title_id: idx for idx, title_id in enumerate(self.titles_df['id'].unique())}
        self.idx_title_map = {idx: title_id for title_id, idx in self.title_idx_map.items()}

        # Create a mapping of title ID to title name
        self.id_to_title = dict(zip(self.titles_df['id'], self.titles_df['title']))

        # Process text data for content-based filtering
        print("Processing text features...")
        stemmer = SnowballStemmer('english')

        def clean_text(text):
            if isinstance(text, str):
                # Remove special characters and lowercase
                text = re.sub('[^a-zA-Z]', ' ', text.lower())
                # Tokenize and stem
                words = [stemmer.stem(word) for word in nltk.word_tokenize(text) if len(word) > 2]
                return ' '.join(words)
            return ''

        # Clean and combine text features
        self.titles_df['clean_description'] = self.titles_df['description'].apply(clean_text)

        # Convert genres list to string for TF-IDF
        self.titles_df['genres_str'] = self.titles_df['genres'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else '')
        # Convert production countries to string
        self.titles_df['countries_str'] = self.titles_df['production_countries'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else '')

        # Process cast and crew data
        print("Processing cast and crew data...")
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

        # Combine all text features
        self.titles_df['combined_features'] = (
                self.titles_df['clean_description'] + ' ' +
                self.titles_df['genres_str'] + ' ' +
                self.titles_df['countries_str'] + ' ' +
                self.titles_df['actors_str'] + ' ' +
                self.titles_df['directors_str']
        )

        self.preprocessed = True
        print("Preprocessing complete.")

    def calculate_combined_score(self, row):

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

        if not self.preprocessed:
            self.preprocess_data()

        print("Building recommendation models...")

        numeric_columns = ['imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']
        for col in numeric_columns:
            if col in self.titles_df.columns:
                self.titles_df[col] = pd.to_numeric(self.titles_df[col], errors='coerce')

        # Calculate combined score if it doesn't exist
        if 'combined_score' not in self.titles_df.columns:
            print("Calculating combined scores for all titles...")
            self.titles_df['combined_score'] = self.titles_df.apply(self.calculate_combined_score, axis=1)

        # 1. Content-based similarity
        print("Building content-based model...")
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.titles_df['combined_features'])
        self.content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # 2. Cast & Crew similarity
        print("Building cast & crew model...")
        # Create actor-title matrix
        actor_title_df = self.credits_df[self.credits_df['role'] == 'ACTOR'].groupby(
            ['person_ID', 'id']).size().reset_index(name='count')

        if len(actor_title_df) > 0:
            actor_title_matrix = csr_matrix(
                (actor_title_df['count'],
                 (actor_title_df['person_ID'].astype('category').cat.codes,
                  [self.title_idx_map.get(tid, 0) for tid in actor_title_df['id']]))
            )
            actor_similarity = cosine_similarity(actor_title_matrix.T, actor_title_matrix.T)
        else:
            actor_similarity = np.zeros((len(self.titles_df), len(self.titles_df)))

        # Create director-title matrix
        director_title_df = self.credits_df[self.credits_df['role'] == 'DIRECTOR'].groupby(
            ['person_ID', 'id']).size().reset_index(name='count')

        if len(director_title_df) > 0:
            director_title_matrix = csr_matrix(
                (director_title_df['count'],
                 (director_title_df['person_ID'].astype('category').cat.codes,
                  [self.title_idx_map.get(tid, 0) for tid in director_title_df['id']]))
            )
            director_similarity = cosine_similarity(director_title_matrix.T, director_title_matrix.T)
            self.actor_director_similarity = 0.7 * actor_similarity + 0.3 * director_similarity
        else:
            self.actor_director_similarity = actor_similarity

        # 3. Popularity model
        print("Building popularity model...")
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
        print("Combining models...")
        self.combined_similarity = 0.6 * self.content_similarity + 0.4 * self.actor_director_similarity

        self.model_built = True
        print("All models built successfully.")

    def _get_recommendations_by_similarity(self, title_id, similarity_matrix, top_n=10):

        idx = self.title_idx_map.get(title_id)
        if idx is None:
            print(f"Title ID {title_id} not found in the dataset.")
            return []
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
        title_indices = [i[0] for i in sim_scores]
        title_ids = [self.idx_title_map[idx] for idx in title_indices]
        recommendations = [
            {
                'id': title_id,
                'title': self.id_to_title.get(title_id),
                'similarity_score': score
            } for title_id, (_, score) in zip(title_ids, sim_scores)
        ]
        return recommendations

    def get_content_recommendations(self, title_id, top_n=10):
        if not self.model_built:
            self.build_models()
        return self._get_recommendations_by_similarity(title_id, self.content_similarity, top_n)

    def get_cast_crew_recommendations(self, title_id, top_n=10):
        if not self.model_built:
            self.build_models()
        return self._get_recommendations_by_similarity(title_id, self.actor_director_similarity, top_n)

    def get_hybrid_recommendations(self, title_id, top_n=10):
        if not self.model_built:
            self.build_models()
        return self._get_recommendations_by_similarity(title_id, self.combined_similarity, top_n)

    def get_popular_in_genre(self, genre, top_n=10):
        if not self.model_built:
            self.build_models()

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
            print(f"No titles found in genre: {genre}")
            return []

        scores = [(idx, self.popularity_scores[idx]) for idx in genre_titles]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
        recommendations = [
            {
                'id': self.titles_df.iloc[idx]['id'],
                'title': self.titles_df.iloc[idx]['title'],
                'popularity_score': score
            } for idx, score in scores
        ]
        return recommendations

    def get_recommendations_for_user(self, liked_title_ids, top_n=20, diversity_factor=0.3):

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

    def get_diverse_recommendations(self, title_id, top_n=10, diversity_factor=0.3):
        if not self.model_built:
            self.build_models()

        # Get initial recommendations based on similarity
        initial_recs = self._get_recommendations_by_similarity(title_id, self.combined_similarity, top_n=50)
        if not initial_recs:
            return []

        # Get indices for recommendations
        rec_indices = [self.title_idx_map[rec['id']] for rec in initial_recs]

        # Extract similarity matrix for these recommendations
        rec_similarity = self.combined_similarity[rec_indices][:, rec_indices]

        # Get combined quality scores for these titles if available
        quality_scores = []
        if 'combined_score' in self.titles_df.columns:
            for rec in initial_recs:
                title_df = self.titles_df[self.titles_df['id'] == rec['id']]
                if not title_df.empty and pd.notna(title_df.iloc[0]['combined_score']):
                    quality_scores.append(title_df.iloc[0]['combined_score'])
                else:
                    quality_scores.append(0)

            # Normalize quality scores
            max_quality = max(quality_scores) if quality_scores and max(quality_scores) > 0 else 1
            quality_scores = [score / max_quality for score in quality_scores]
        else:
            # If no combined scores, just use 0.5 as default
            quality_scores = [0.5] * len(initial_recs)

        # Add quality score to initial recommendations
        for i, rec in enumerate(initial_recs):
            rec['quality_score'] = quality_scores[i]

        # Select diverse recommendations
        selected_indices = []
        selected_recs = []

        # Start with the first recommendation
        selected_indices.append(0)
        selected_recs.append(initial_recs[0])

        # Keep selecting until we have enough or run out of candidates
        while len(selected_recs) < top_n and len(selected_indices) < len(rec_indices):
            max_score = -float('inf')
            next_idx = -1

            # Evaluate each candidate recommendation
            for i in range(len(rec_indices)):
                if i not in selected_indices:
                    # Calculate dissimilarity to already selected items
                    sim_to_selected = np.mean([rec_similarity[i, j] for j in selected_indices])
                    div_score = (1 - sim_to_selected) * diversity_factor

                    # Calculate combined score with similarity, diversity and quality
                    recommendation_score = (
                            initial_recs[i]['similarity_score'] * (1 - diversity_factor) * 0.7 +  # Similarity
                            div_score * 0.2 +  # Diversity
                            initial_recs[i]['quality_score'] * 0.1  # Quality
                    )

                    # Keep track of highest scoring candidate
                    if recommendation_score > max_score:
                        max_score = recommendation_score
                        next_idx = i

            if next_idx != -1:
                selected_indices.append(next_idx)
                selected_recs.append(initial_recs[next_idx])
            else:
                break

        return selected_recs

    def explain_recommendation(self, source_title_id, rec_title_id):
        if not self.model_built:
            self.build_models()

        source_idx = self.title_idx_map.get(source_title_id)
        rec_idx = self.title_idx_map.get(rec_title_id)
        if source_idx is None or rec_idx is None:
            return {"error": "One or both title IDs not found"}

        source_title = self.titles_df.iloc[source_idx]
        rec_title = self.titles_df.iloc[rec_idx]

        content_sim = self.content_similarity[source_idx, rec_idx]
        cast_crew_sim = self.actor_director_similarity[source_idx, rec_idx]

        source_genres = source_title['genres'] if isinstance(source_title['genres'], list) else []
        rec_genres = rec_title['genres'] if isinstance(rec_title['genres'], list) else []
        shared_genres = set(source_genres).intersection(set(rec_genres))

        source_actors = self.title_actors.get(source_title_id, [])
        rec_actors = self.title_actors.get(rec_title_id, [])
        shared_actors = set(source_actors).intersection(set(rec_actors))

        source_directors = self.title_directors.get(source_title_id, [])
        rec_directors = self.title_directors.get(rec_title_id, [])
        shared_directors = set(source_directors).intersection(set(rec_directors))

        # Get quality metrics
        source_quality = source_title.get('combined_score', None) if 'combined_score' in source_title else None
        rec_quality = rec_title.get('combined_score', None) if 'combined_score' in rec_title else None

        explanation = {
            "source_title": source_title['title'],
            "recommended_title": rec_title['title'],
            "overall_similarity": self.combined_similarity[source_idx, rec_idx],
            "content_similarity": content_sim,
            "cast_crew_similarity": cast_crew_sim,
            "shared_genres": list(shared_genres),
            "shared_actors": list(shared_actors)[:5],
            "shared_directors": list(shared_directors),
            "show_type_match": source_title['show_type'] == rec_title['show_type'],
            "source_quality_score": source_quality,
            "recommended_quality_score": rec_quality
        }
        return explanation