# Bismillahi-r-Rahmani-r-Rahim
#
# Class to represent an experiment in lexical entailment


from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold

from datetime import datetime
import hashlib
from baseline import tools
import logging

class EntailmentExperiment(object):
    def __init__(self, dataset, classifier, num_folds):
        self.dataset = dataset
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
        
        return confusion, time, 'Kfold test'

class EntailmentExperimentHeldOut(EntailmentExperiment):

    def __init__(self,dataset,classifier,num_folds,blesspath):
        self.dataset = dataset
        self.classifier = classifier
        self.num_folds = num_folds
        self.annotation = tools.annotate(self.num_folds,blesspath)  #dictionary of bless concepts annotated with cross-validation fold


    def runFold(self,fold):
        train_indices=[]
        test_indices=[]
        for i,[w1,w2,sc] in enumerate(self.dataset):
            if self.annotation.get(w1,-1)==fold or self.annotation.get(w2,-1)==fold:
                test_indices.append(i)
            else:
                train_indices.append(i)

        message="Fold "+str(fold)+", training set size: "+str(len(train_indices))+", test set size: "+str(len(test_indices))
        logging.info(message)
        #print message

        train = [self.dataset[i] for i in train_indices]
        test = [self.dataset[i] for i in test_indices]

        start = datetime.now()
        self.classifier.fit(train)
        time = datetime.now() - start
        results = self.classifier.predict(test)

        test_target = [x[2] for x in test]
        confusion = confusion_matrix(test_target, results)

        return confusion, time, 'Held out test'

class EntailmentExperimentHeldOutStrict(EntailmentExperiment):

    def __init__(self,dataset,classifier,num_folds,blesspath):
        self.dataset = dataset
        self.classifier = classifier
        self.num_folds = num_folds
        self.annotation = tools.annotate(self.num_folds,blesspath)  #dictionary of bless concepts annotated with cross-validation fold


    def runFold(self,fold):
        train_indices=[]
        test_indices=[]
        test_concepts=[]
        for i,[w1,w2,sc] in enumerate(self.dataset):
            if self.annotation.get(w1,-1)==fold or self.annotation.get(w2,-1)==fold:
                test_indices.append(i)
                test_concepts.append(w1)
                test_concepts.append(w2)
        for i,[w1,w2,sc] in enumerate(self.dataset):
            if w1 not in test_concepts and w2 not in test_concepts:
                train_indices.append(i)

        message="Fold "+str(fold)+", training set size: "+str(len(train_indices))+", test set size: "+str(len(test_indices))
        logging.info(message)
        print message

        train = [self.dataset[i] for i in train_indices]
        test = [self.dataset[i] for i in test_indices]

        start = datetime.now()
        self.classifier.fit(train)
        time = datetime.now() - start
        results = self.classifier.predict(test)

        test_target = [x[2] for x in test]
        confusion = confusion_matrix(test_target, results)

        return confusion, time, 'Strict held out test'


class EntailmentExperimentTrainTest(EntailmentExperiment):

    def __init__(self,trainset,classifier,testset):
        self.num_folds=1
        self.dataset=trainset
        self.classifier=classifier
        self.testset=testset


    def runFold(self,fold):
        start = datetime.now()
        self.classifier.fit(self.dataset)
        time = datetime.now()-start
        results=self.classifier.predict(self.testset)
        test_target=[x[2] for x in self.testset]
        confusion = confusion_matrix(test_target,results)
        return confusion, time, 'Single Train-Test run'
