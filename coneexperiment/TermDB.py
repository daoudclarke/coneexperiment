# Bismillahi-r-Rahmani-r-Rahim
#
# Database of term vectors

import sqlite3
import sys
import numpy as np
import os
import json

#sys.setdefaultencoding('utf_8')

def create_db(file_path):
    with open(file_path) as data:
        db_path = file_path + '.db'
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS termvector
                (
                  term TEXT PRIMARY KEY,
                  vector TEXT          
                )
                """)
            for term, vector in dependencies(data):
                try:
                    # values = [unicode(term).encode('utf-8'),
                    #           unicode(json.dumps(vector)).encode('utf-8')]
                    values = [str.decode(term, 'utf-8'),
                              json.dumps(vector)]
                except UnicodeDecodeError as e:
                    print unicode.decode(unicode(term), 'utf-8')
                    print "Exception for term", term, vector
                    raise

                cursor.execute("""
                  INSERT INTO termvector (term, vector)
                  VALUES (?, ?)
                               """, values)


    
def get_features(line):
    split = line.split()
    word = split[0]
    features = {}
    for i in range(1, len(split),2):
        #print split[i], split[i+1]
        features[split[i]] = float(split[i+1])
    return word, features

def dependencies(data):
    for line in data:
        word, features = get_features(line)
        features = {x[0]:x[1] for x in features.iteritems()
                    if not x[0].startswith('T:')
                    and not x[0].startswith('__')}
        yield word, features

    

class TermDB(object):
    def __init__(self, db_path):
        self.db_path = db_path


    def __getitem__(self, key):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
               SELECT vector FROM termvector
               WHERE term=?
               """, (key,))
            try:
                vector = json.loads(
                    [x for x in cursor][0][0])
                return vector
            except IndexError:
                return {}

if __name__ == "__main__":
    create_db(sys.argv[1])
