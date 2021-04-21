import tensorflow as tf
from feature_extractor_mobilenet import FeatureExtractor

fe = FeatureExtractor()

print(fe.model.summary())