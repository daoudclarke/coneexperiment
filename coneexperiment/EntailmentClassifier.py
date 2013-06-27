# Bismillahi-r-Rahmani-r-Rahim
#
# Classifier that operates on pairs of terms

from sklearn.feature_extraction import DictVectorizer

class EntailmentClassifier:
    def __init__(self, classifier, termVectors):
        self.termVectors = termVectors
        self.classifier = classifier

    def fit(self, pairs):
        """Train classifier based on a sequence of (word1, word2, entail) tuples.
        The value of entail is True if word1 entails word2 or False otherwise."""
        data = self.value_map(pairs)
        target = [p[2] for p in pairs]
        self.classifier.fit(data, target)

    def predict(self, pairs):
        "Predict whether entailment holds for a sequence of (word1, word2) tuples."
        data = self.value_map(pairs)
        return self.classifier.predict(data)

    def value_map(self, pairs):
        return [self.termVectors[p[1]] - self.termVectors[p[0]]
                for p in pairs]
        
