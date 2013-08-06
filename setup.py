from setuptools import setup
from disttest import test

# Import this to prevent spurious error: info('process shutting down')
from multiprocessing import util

setup(name='coneexperiment',
      version='0.1',
      description='Experiments on lexical entailment using cones',
      #url='http://github.com/storborg/funniest',
      author='Daoud Clarke',
      #author_email='flyingcircus@example.com',
      license='MIT',
      packages=['coneexperiment','baseline'],
      cmdclass = {'test': test},
      options = {'test' : {'test_dir':['test']}}
      #zip_safe=False
      )
