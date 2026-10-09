"""
Layer 3 — optional local classifier.

This entire layer is optional and removable. The pipeline works without it.
Remove by not including 'layer3_classifier' in RouterConfig.layers.
"""
from .classifier import ClassifierResult, LocalClassifier
from .layer import ClassifierRoutingLayer

__all__ = ["LocalClassifier", "ClassifierResult", "ClassifierRoutingLayer"]
