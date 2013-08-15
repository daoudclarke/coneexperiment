#!/bin/bash
rm log/experiments.log; python -u coneexperiment/EntailmentSuite.py entailment.cfg -n 1
