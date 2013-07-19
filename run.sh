#!/bin/bash
rm -r results/entailment-test/; rm log/experiments.log; python coneexperiment/EntailmentSuite.py entailment.cfg -n 1
