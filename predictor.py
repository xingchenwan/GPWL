class Predictor:

    def __init__(self, ss_type=None, encoding_type=None):
        self.ss_type = ss_type
        self.encoding_type = encoding_type

    def fit(self, xtrain, ytrain, info=None):
        """
        This can be called any number of times during the NAS algorithm.
        input: list of architectures, list of architecture accuracies
        output: none
        """
        pass

    def predict(self, xtest, info):
        """
        This can be called any number of times during the NAS algorithm.
        inputs: list of architectures,
                info about the architectures (e.g., training data up to 20 epochs)
        output: predictions for the architectures
        """
        pass

    def set_hyperparams(self, hyperparams):
        self.hyperparams = hyperparams

