from pydantic import BaseModel

class MovieRequest(BaseModel):
    description: str

class ClusterPrediction(BaseModel):
    cluster: int
    cluster_name: str
    description: str