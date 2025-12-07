"""Model factory for creating news recommendation models."""
from typing import Type

import torch.nn as nn

from config import NRMSbertConfig
from model.base import BaseNewsRecommendationModel
from model.ColBERT import ColBERTNewsRecommendationModel
from model.ColBERT.naml_variant import ColBERTNAMLModel
from model.ColBERT.lstur_variant import ColBERTLSTURModel
from model.NAMLbert import NAMLbert
from model.NRMSbert import NRMSbert
from model.LSTURbert import LSTURbert


_MODEL_REGISTRY: dict[str, Type[BaseNewsRecommendationModel]] = {
    'NRMSbert': NRMSbert,
    'NAMLbert': NAMLbert,
    'LSTURbert': LSTURbert,
    'ColBERT': ColBERTNewsRecommendationModel,
    'ColBERT-NAML': ColBERTNAMLModel,
    'ColBERT-LSTUR': ColBERTLSTURModel,
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
    
    # Handle ColBERT variant selection
    if model_type == 'ColBERT':
        variant = getattr(config, 'colbert_variant', 'nrms')
        if variant == 'naml':
            model_type = 'ColBERT-NAML'
        elif variant == 'lstur':
            model_type = 'ColBERT-LSTUR'
        # else: use default ColBERT (NRMS variant)
    
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

