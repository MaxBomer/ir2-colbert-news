"""Model factory for creating news recommendation models."""
from typing import Type

import torch.nn as nn

from config import NRMSbertConfig
from model.base import BaseNewsRecommendationModel
from model.ColBERT import ColBERTNewsRecommendationModel
from model.NRMSbert import NRMSbert


_MODEL_REGISTRY: dict[str, Type[BaseNewsRecommendationModel]] = {
    'NRMSbert': NRMSbert,
    'ColBERT': ColBERTNewsRecommendationModel,
}


def create_model(config: NRMSbertConfig) -> BaseNewsRecommendationModel:
    """Create a model instance based on config.
    
    Args:
        config: Configuration object
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model_type is not supported
    """
    model_type = config.model_type
    
    if model_type not in _MODEL_REGISTRY:
        available = ', '.join(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: {available}"
        )
    
    model_class = _MODEL_REGISTRY[model_type]
    return model_class(config)


def register_model(name: str, model_class: Type[BaseNewsRecommendationModel]) -> None:
    """Register a new model type.
    
    Args:
        name: Model type name
        model_class: Model class (must inherit from BaseNewsRecommendationModel)
    """
    if not issubclass(model_class, BaseNewsRecommendationModel):
        raise TypeError(
            f"Model class must inherit from BaseNewsRecommendationModel, "
            f"got {model_class.__name__}"
        )
    _MODEL_REGISTRY[name] = model_class


def list_available_models() -> list[str]:
    """List all available model types.
    
    Returns:
        List of model type names
    """
    return list(_MODEL_REGISTRY.keys())

