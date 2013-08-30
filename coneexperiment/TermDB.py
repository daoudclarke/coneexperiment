# Bismillahi-r-Rahmani-r-Rahim
#
# Database of term vectors

import sqlite3
import sys
import numpy as np
import os
import json
import logging
import codecs

    
def get_features(line):
    split = line.split()
    features = {}
    i = 1
    while i < len(split):
        try:
            features[split[i]] = float(split[i+1])
        except ValueError:
            logging.warning('Invalid term found in data file: %s',
                            str(split[i:i+3]))
            i += 3
            continue
        i += 2
    return features

def get_dependencies(line):
    features = get_features(line)
    features = {x[0]:x[1] for x in features.iteritems()
                if not x[0].startswith('T:')
                and not x[0].startswith('__')}
    return features
    
class TermPosDB(object):
    def __init__(self, db_path, pos):
        self.db_path = db_path
        self.pos = pos

    def load(self, terms):
        """Load the terms into the database. Must be called before use.
        Pass in the list of terms that will be needed."""
        self.term_vectors = {}
        terms = set(terms)
        logging.debug("Loading terms: %s", str(terms))
        with codecs.open(self.db_path,encoding='utf-8') as db:
            for line in db:
                term = line.split('/')[0]
                if term in terms:
                    features = get_dependencies(line)
                    logging.debug('DB getitem: accessed term %s with POS %s from DB',
                                  term, self.pos)
                    self.term_vectors[term] = features
                    return features

    def __getitem__(self, key):
        if key in self.term_vectors:
            logging.debug('DB getitem: accessed term %s with POS %s from cache',
                          key, self.pos)
            return self.term_vectors[key]
        logging.debug('DB getitem: term %s with POS %s missing in DB',
                      key, self.pos)
        return {}


class TermDB(object):
    def __init__(self, db_path):
        self.nouns = TermPosDB(db_path, 'N')

if __name__ == "__main__":
    logging.basicConfig(filename='log/termdb.log',
                        level=logging.DEBUG,
                        format='%(asctime)s %(process)d %(levelname)s %(message)s')
    create_db(sys.argv[1])
