import torch
import torch.nn as nn
from model.base import BaseNewsRecommendationModel


class RandomNewsRecommendationModel(BaseNewsRecommendationModel):

    def __init__(self, config) -> None:
        super().__init__()
        self.embedding_dim = 128

    def forward(self, candidate_news, clicked_news, clicked_news_mask):

        # Look at how many samples (impressions) are in the batch
        batch_size = candidate_news[0]["title"].shape[0]

        # Look at how many candidate articles need scoring
        num_candidates = len(candidate_news)

        # Assign a random score for every news conadidate for every impression 
        return torch.rand(batch_size, num_candidates)

    def get_news_vector(self, news):

        # For every news vector, just return a random vector
        batch_size = news["title"].shape[0]
        return torch.rand(batch_size, self.embedding_dim)

    def get_user_vector(self, clicked_news_vector):

        # For every user vector, just return a random vector
        batch_size = clicked_news_vector.shape[0]
        return torch.rand(batch_size, self.embedding_dim)

    def get_prediction(self, news_vector, user_vector):

        # Given a user and several candidate news vectors, return a random score per candidate
        num_candidates = news_vector.shape[0]
        return torch.rand(num_candidates)
