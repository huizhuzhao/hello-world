## The process of self-taught learning
- preprocessing of unlabeled data-'(X_unlabeled, )', and save away the preprocessing parameters for latter use
- training a sparse autoencoder on the above preprocessed unlabeled data
- preprocessing the labeled data-(X_labeled, Y) (suppose the amount of labeled data is small, and the preprocessing is done with
    the already saved away preprocessing parameters), then using the trained autoencoder to get the new representations
    of the labeled data-(X_activation, Y) (i.e. the activations in the hidden layer)
- training the new labeled data-(X_activation, Y) with supervised learning algorithm, like SVM, logistic regression, etc. to obtain a function that makes predictions on the Y values.
