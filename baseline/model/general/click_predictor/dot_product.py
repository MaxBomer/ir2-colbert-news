import torch


class DotProductClickPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super(DotProductClickPredictor, self).__init__()

    def forward(self, candidate_news_vector: torch.Tensor, user_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidate_news_vector: batch_size, candidate_size, X
            user_vector: batch_size, X
        Returns:
            (shape): batch_size
        """
        # batch_size, candidate_size
        probability = torch.bmm(candidate_news_vector,
                                user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        return probability
