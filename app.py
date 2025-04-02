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

# (NLTK and NetflixRecommender setup code remains unchanged...)
# ... [Your NLTK and NetflixRecommender setup code here] ...

# Sidebar for configuration
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/7a/Logonetflix.png", width=200)
    st.markdown("### Recommendation Settings")

    # Recommendation Type Selector
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

    # Other configuration options
    diversity_factor = st.slider("Diversity factor", 0.0, 1.0, 0.3, 0.1,
                                 help="Higher values prioritize diversity over similarity")
    num_recommendations = st.slider("Number of recommendations", 5, 20, 10)
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

# Main content (outside the sidebar)
if st.session_state.loading:
    with st.spinner(st.session_state.loading_message):
        if load_data_and_build_models():
            st.success("Models built successfully!")
            st.rerun()
elif not st.session_state.model_built:
    st.error("Failed to load recommendation models. Please refresh the page to try again.")
else:
    # Initialize selection mode if not yet set
    if 'recommendation_mode' not in st.session_state:
        st.session_state.recommendation_mode = False
        st.session_state.selected_titles = []

    # Check if a detail modal should be shown
    if st.session_state.show_detail_modal and st.session_state.detail_title_id:
        show_title_details(st.session_state.detail_title_id)

    # INITIAL SELECTION MODE: Display movie selection UI in the main area
    if not st.session_state.recommendation_mode:
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
                                st.markdown(f"**{title.get('show_type', 'N/A')} ({title.get('release_year', 'N/A')})**")
                                genres = title['genres']
                                if isinstance(genres, list) and genres:
                                    st.markdown(f"**Genres:** {', '.join(genres[:3])}")
                                elif isinstance(genres, str):
                                    try:
                                        genres_list = json.loads(genres.replace("'", '"'))
                                        st.markdown(f"**Genres:** {', '.join(genres_list[:3])}")
                                    except Exception:
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
                                    f"**IMDB Score:** {imdb_score} <span class='combined-score-badge'>Combined Score: {combined_score_formatted}</span>",
                                    unsafe_allow_html=True
                                )
                                if st.button("View Details", key=f"view_{title['id']}"):
                                    st.session_state.show_detail_modal = True
                                    st.session_state.detail_title_id = title['id']
                                    st.rerun()
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
            st.markdown("### Select a Genre")
            genres_per_row = 5
            genre_rows = [unique_genres[i:i + genres_per_row] for i in range(0, len(unique_genres), genres_per_row)]
            for row in genre_rows:
                cols = st.columns(genres_per_row)
                for i, genre in enumerate(row):
                    with cols[i]:
                        if st.button(genre, key=f"genre_{genre}", help=f"Show popular titles in {genre} genre"):
                            st.session_state.selected_genre = genre
                            st.rerun()
            if st.session_state.selected_genre:
                st.markdown(f"### Popular Titles in {st.session_state.selected_genre}")
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
                                title_info = st.session_state.titles_df[st.session_state.titles_df['id'] == rec['id']]
                                if not title_info.empty:
                                    title_details = title_info.iloc[0]
                                    with cols[j]:
                                        with st.container():
                                            st.markdown(f"### {rec['title']}")
                                            st.markdown(
                                                f"**{title_details.get('show_type', 'N/A')} ({title_details.get('release_year', 'N/A')})**")
                                            genres = title_details.get('genres', [])
                                            if isinstance(genres, list) and genres:
                                                st.markdown(f"**Genres:** {', '.join(genres[:3])}")
                                            elif isinstance(genres, str):
                                                try:
                                                    genres_list = json.loads(genres.replace("'", '"'))
                                                    st.markdown(f"**Genres:** {', '.join(genres_list[:3])}")
                                                except Exception:
                                                    st.markdown(f"**Genres:** {genres}")
                                            description = title_details.get('description', '')
                                            if description:
                                                st.markdown(f"<div class='truncate-text'>{description}</div>",
                                                            unsafe_allow_html=True)
                                            imdb_score = title_details['imdb_score'] if pd.notna(
                                                title_details['imdb_score']) else "N/A"
                                            combined_score = title_details['combined_score'] if pd.notna(
                                                title_details['combined_score']) else "N/A"
                                            combined_score_formatted = f"{combined_score:.2f}" if isinstance(
                                                combined_score, float) else combined_score
                                            st.markdown(
                                                f"**IMDB Score:** {imdb_score} <span class='combined-score-badge'>Combined Score: {combined_score_formatted}</span>",
                                                unsafe_allow_html=True
                                            )
                                            if st.button("View Details", key=f"view_genre_{rec['id']}"):
                                                st.session_state.show_detail_modal = True
                                                st.session_state.detail_title_id = rec['id']
                                                st.rerun()
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

    # RECOMMENDATION MODE
    else:
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
                    st.warning(f"Couldn't get recommendations for title {title_id}: {str(e)}")
            filtered_recs = {}
            for rec in all_recommendations:
                if rec['id'] not in liked_ids and rec['id'] not in filtered_recs:
                    filtered_recs[rec['id']] = rec
            recommendations = sorted(filtered_recs.values(),
                                     key=lambda x: x['similarity_score'],
                                     reverse=True)[:num_recommendations]
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
                    st.warning(f"Couldn't get recommendations for title {title_id}: {str(e)}")
            filtered_recs = {}
            for rec in all_recommendations:
                if rec['id'] not in liked_ids and rec['id'] not in filtered_recs:
                    filtered_recs[rec['id']] = rec
            recommendations = sorted(filtered_recs.values(),
                                     key=lambda x: x['similarity_score'],
                                     reverse=True)[:num_recommendations]
        else:
            st.markdown("<div class='recommendation-type-selector'>", unsafe_allow_html=True)
            st.markdown("🔄 **Using Hybrid recommendations** - balancing content, cast & crew, and quality metrics")
            st.markdown("</div>", unsafe_allow_html=True)
            try:
                recommendations = st.session_state.recommender.get_recommendations_for_user(
                    liked_ids, top_n=num_recommendations, diversity_factor=diversity_factor)
            except Exception as e:
                st.error(f"Error getting hybrid recommendations: {str(e)}")
                recommendations = []
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
                        rec_info = st.session_state.titles_df[st.session_state.titles_df['id'] == rec['id']]
                        if not rec_info.empty:
                            rec_details = rec_info.iloc[0]
                            with cols[j]:
                                with st.container():
                                    st.markdown(f"### {idx + 1}. {rec['title']}")
                                    st.markdown(
                                        f"**{rec_details.get('show_type', 'N/A')} ({rec_details.get('release_year', 'N/A')})**")
                                    genres = rec_details.get('genres', [])
                                    if isinstance(genres, list) and genres:
                                        st.markdown(f"**Genres:** {', '.join(genres[:3])}")
                                    elif isinstance(genres, str):
                                        try:
                                            genres_list = json.loads(genres.replace("'", '"'))
                                            st.markdown(f"**Genres:** {', '.join(genres_list[:3])}")
                                        except Exception:
                                            st.markdown(f"**Genres:** {genres}")
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
                                        st.session_state.show_detail_modal = True
                                        st.session_state.detail_title_id = rec['id']
                                        st.rerun()

st.markdown("<div class='footer'>Netflix Advanced Recommender System - Created with Streamlit</div>",
            unsafe_allow_html=True)

