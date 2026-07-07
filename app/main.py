from fastapi import FastAPI, HTTPException

from app.models import RecommendationRequest

from app.dependencies import engine


app = FastAPI(

    title="Movie Recommendation API",

    version="1.0"

)

@app.get("/")
def root():

    return {

        "message":"Movie Recommendation API"

    }
    
@app.get("/health")
def health():

    return {

        "status":"running"

    }
    
@app.get("/search")
def search_movie(query: str):

    movies = engine.search_movie(query)

    return movies.to_dict(
        orient="records"
    )
    
@app.post("/recommend")
def recommend(request: RecommendationRequest):

    try:

        recommendations = engine.recommend(

            movie_name=request.movie_name,

            top_k=request.top_k,

            min_rating=request.min_rating,

            min_votes=request.min_votes,

            after_year=request.after_year,

            language=request.language

        )

        return recommendations.to_dict(

            orient="records"

        )

    except ValueError as e:

        raise HTTPException(

            status_code=404,

            detail=str(e)

        )