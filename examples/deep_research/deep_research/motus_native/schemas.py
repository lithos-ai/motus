from typing import List

from pydantic import BaseModel


# Structure the output of planning task
class TopicPlan(BaseModel):
    name: str
    search_queries: List[str]


class ResearchPlan(BaseModel):
    topics: List[TopicPlan]
