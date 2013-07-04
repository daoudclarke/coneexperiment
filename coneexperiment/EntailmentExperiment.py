# Bismillahi-r-Rahmani-r-Rahim
#
# Class to represent an experiment in lexical entailment


from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold

from datetime import datetime
import hashlib


class EntailmentExperiment(object):
    def __init__(self, dataset, vectors, classifier, num_folds):
        self.dataset = dataset
        self.vectors = vectors
        self.classifier = classifier
        self.num_folds = num_folds

    def runFold(self, fold):
        seed = int(hashlib.sha1(str(self.dataset[:100])).hexdigest()[:7], 16)
        cv = KFold(n_folds = self.num_folds, n = len(self.dataset),
                   shuffle = True, random_state = seed)
        train_indices, test_indices = list(cv)[fold]
        train = [self.dataset[i] for i in train_indices]
        test = [self.dataset[i] for i in test_indices]

        start = datetime.now()
        self.classifier.fit(train)
        time = datetime.now() - start
        results = self.classifier.predict(test)

        test_target = [x[2] for x in test]
        confusion = confusion_matrix(test_target, results)
        
        return confusion, 1002.3, 'Dummy test'
