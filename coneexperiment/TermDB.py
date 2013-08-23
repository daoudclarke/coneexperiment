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

#sys.setdefaultencoding('utf_8')

def create_db(file_path):
    with open(file_path) as data:
        db_path = file_path + '.db'
        try:
            os.remove(db_path)
        except OSError:
            logging.info('Removing existing database and rebuilding')            
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS termvector")
            cursor.execute("""
                CREATE TABLE termvector
                (
                  term TEXT,
                  pos TEXT,
                  vector TEXT,
                  PRIMARY KEY (term, pos)
                )
                """)
            for term, vector in dependencies(data):
                decoded_term = str.decode(term, 'utf-8')
                values = (decoded_term.split('/') +
                          [json.dumps(vector)])
                if len(values) != 3:
                    logging.warn("Skipping term '%s': Should be exactly one '/' character",
                                 decoded_term)
                    continue

                cursor.execute("""
                  INSERT INTO termvector (term, pos, vector)
                  VALUES (?, ?, ?)
                               """, values)


    
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
        self.cache = {}

    def __getitem__(self, key):
        if key in self.cache:
            logging.debug('DB getitem: accessed term %s with POS %s from cache',
                          key, self.pos)
            return self.cache[key]

        with codecs.open(self.db_path,encoding='utf-8') as db:
            for line in db:
                if line.startswith(key + '/' + self.pos):
                    features = get_dependencies(line)
                    logging.debug('DB getitem: accessed term %s with POS %s from DB',
                                  key, self.pos)
                    self.cache[key] = features
                    return features

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
