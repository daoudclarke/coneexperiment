# Bismillahi-r-Rahmani-r-Rahim
#
# Utils for unit tests

from numpy import random
from string import ascii_lowercase

def randomWord():
    return ''.join([ascii_lowercase[random.randint(len(ascii_lowercase))]
                    for i in range(5)])
