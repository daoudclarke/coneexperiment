# Bismillahi-r-Rahmani-r-Rahim
#
# Utils for unit tests

from numpy import random
from string import ascii_lowercase

def randomWord():
    return ''.join([ascii_lowercase[random.randint(len(ascii_lowercase))]
                    for i in range(5)])

def testData():
    unzipped = [('cat','dog', 'banana'),
                ('animal', 'mosquito', 'fruit'),
                tuple(random.randint(0,2,3) == 0)]
    test_words = [randomWord() for j in range(6)]
    words = unzipped[0] + unzipped[1] + tuple(test_words)
    word_vectors = [random.random(10) for j in range(6)]
    vectors = {words[j]:word_vectors[j%6] for j in range(len(words))}
    data = zip(*unzipped)
    test_data = zip(test_words[:3], test_words[3:6], unzipped[2])
    return data, test_data, vectors
