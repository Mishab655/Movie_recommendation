from pydantic import BaseModel


class RecommendationRequest(BaseModel):

    movie_name: str

    top_k: int = 10

    min_rating: float | None = None

    min_votes: int | None = None

    after_year: int | None = None

    language: str | None = None