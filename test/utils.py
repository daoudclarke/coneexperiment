# Bismillahi-r-Rahmani-r-Rahim
#
# Utils for unit tests

from numpy import random
from string import ascii_lowercase

class TestTermDb(object):
    def __init__(self, nouns):
        self.nouns = nouns

def randomWord():
    return ''.join([ascii_lowercase[random.randint(len(ascii_lowercase))]
                    for i in range(5)])

def testData():
    unzipped = [('cat','dog', 'banana', 'orange', 'butterfly', 'plug'),
                ('animal', 'mosquito', 'fruit', 'table', 'wing', 'cup'),
                tuple(random.randint(0,2,6) == 0)]
    data = zip(*unzipped)

    test_words = [randomWord() for j in range(12)]
    test_data = zip(test_words[:6], test_words[6:], unzipped[2])

    words = unzipped[0] + unzipped[1] + tuple(test_words)
    word_vectors = [{randomWord():random.randint(100) for i in range(10)}
                    for j in range(len(test_words))]
    vectors = {words[j]:word_vectors[j%12] for j in range(len(words))}

    return data, test_data, TestTermDb(vectors)
