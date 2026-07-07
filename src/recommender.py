import pandas as pd


class RecommendationEngine:

    def __init__(self, movies_df, similarity_matrix):

        self.movies = movies_df.copy()

        self.similarity_matrix = similarity_matrix

        # Convert release_date once during initialization
        self.movies["release_date"] = pd.to_datetime(
            self.movies["release_date"],
            errors="coerce"
        )

        self.movies["release_year"] = (
            self.movies["release_date"].dt.year
        )

        # Create movie-title → index mapping
        self.indices = pd.Series(
            self.movies.index,
            index=self.movies["original_title"]
        ).drop_duplicates()

    # ----------------------------------------------------
    # Utility Methods
    # ----------------------------------------------------

    def movie_exists(self, movie_name):

        return movie_name in self.indices

    def get_movie_index(self, movie_name):

        if not self.movie_exists(movie_name):

            raise ValueError(
                f"Movie '{movie_name}' not found."
            )

        return self.indices[movie_name]

    def search_movie(self, keyword):

        return self.movies[
            self.movies["original_title"]
            .str.contains(
                keyword,
                case=False,
                na=False
            )
        ][["original_title"]]

    # ----------------------------------------------------
    # Recommendation
    # ----------------------------------------------------

    def recommend(

        self,

        movie_name,

        top_k=10,

        min_rating=None,

        min_votes=None,

        after_year=None,

        language=None

    ):

        movie_index = self.get_movie_index(movie_name)

        similarity_scores = list(
            enumerate(
                self.similarity_matrix[movie_index]
            )
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        # Remove the queried movie
        similarity_scores = similarity_scores[1:]

        movie_indices = [
            score[0]
            for score in similarity_scores
        ]

        recommendations = self.movies.iloc[
            movie_indices
        ].copy()

        recommendations["similarity"] = [
            score[1]
            for score in similarity_scores
        ]

        # -----------------------------
        # Apply Filters
        # -----------------------------

        if min_rating is not None:

            recommendations = recommendations[
                recommendations["vote_average"] >= min_rating
            ]

        if min_votes is not None:

            recommendations = recommendations[
                recommendations["vote_count"] >= min_votes
            ]

        if after_year is not None:

            recommendations = recommendations[
                recommendations["release_year"] >= after_year
            ]

        if language is not None:

            recommendations = recommendations[
                recommendations["original_language"] == language
            ]

        # -----------------------------
        # Sort Again
        # -----------------------------

        recommendations = recommendations.sort_values(
            "similarity",
            ascending=False
        )

        # -----------------------------
        # Return Useful Columns
        # -----------------------------

        return recommendations[
            [
                "original_title",
                "release_year",
                "vote_average",
                "vote_count",
                "popularity",
                "similarity"
            ]
        ].head(top_k)