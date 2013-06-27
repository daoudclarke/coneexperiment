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
        data, target = self.fit_transform(pairs)
        self.classifier.fit(data, target)

    def predict(self, pairs):
        "Predict whether entailment holds for a sequence of (word1, word2) tuples."
        data = self.transform(pairs)
        return self.classifier.predict(data)

    def fit_transform(self, pairs):
        print pairs
        values = [{'word':p[0] + '_' + p[1]} for p in pairs]
        self.vec = DictVectorizer()
        vectors = self.vec.fit_transform(values).toarray()
        target = [p[2] for p in pairs]
        return vectors, target

    def transform(self, pairs):
        values = [{'word':p[0] + '_' + p[1]} for p in pairs]
        return self.vec.transform(values).toarray()
