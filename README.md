# How-to-estimate-required-sample-size-for-model-training
This is an implementation trying to answer the question "how many images will we need to train a good enough machine learning model?" modeling the relationship between training set size and model accuracy.

In most cases, a small set of samples is available, and we can use it to model the relationship between training data size and model performance. Such a model can be used to estimate the optimal number of images needed to arrive at a sample size that would achieve the required model performance.

# Sources
- (Keras example: https://keras.io/examples/keras_recipes/sample_size_estimate/)
- Original article: Sample-Size Determination Methodologies for Machine Learning in Medical Imaging Research: A Systematic Review : https://www.researchgate.net/publication/335779941_Sample-Size_Determination_Methodologies_for_Machine_Learning_in_Medical_Imaging_Research_A_Systematic_Review
