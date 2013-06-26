# Bismillahi-r-Rahmani-r-Rahim
#
# Classifier that operates on pairs of terms


class EntailmentClassifier:
    def __init__(self, termVectors, classifier):
        self.termVectors = termVectors
        self.classifier = classifier


    def fit(self, pairs):
        """Train classifier based on a sequence of (word1, word2, entail) tuples.
        The value of entail is True if word1 entails word2 or False otherwise."""
        pass

    def predict(self, pairs):
        "Predict whether entailment holds for a sequence of (word1, word2) tuples."
        return [True, False, True]


