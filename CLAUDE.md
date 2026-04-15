# Project Overview

This is a working directory to implement the conformal prediction algorithm that we develop, named AdaptedCAFHT. 

We verify the performances in three situations: (1) fully synthetic data; (2) finance data; (3) medical data (developing).

# File Structure

ts_generator.py: generates fully synthetic AR(1) processes.

adaptive_conformal.py: baseline algorithm of standard conformal prediction.

algorithm.py: implementation of AdaptedCAFHT algorithm. 

test_conformal.py: tests the performances of our algorithm and baseline algorithm on synthetic data.

multi_seed_experiments.py: runs fully synthetic tests for many seeds.

finance_data.py: prepares the finance data.

finance_conformal.py: runs experiment of AdaptedCAFHT on finance data.

medical_data.py: loads and summarizes the sepsis ICU trajectory dataset (sepsis_experiment_data.pkl).

medical_conformal.py: runs experiment of AdaptedCAFHT on sepsis ICU data (Heart Rate prediction).

# Rules of Coding

1. Write very good instructions/information before codes in each file, just as what finance_conformal.py did. 

You should at least include QUICK START (example usage in terminal), FULL OPTIONS (parameters to choose, and their default values), HOW IT WORKS (explain the logic of the main for-loop, what you do at each step). 

Write one-line comments before each function to explain the uses/input/output of the function.

2. Everything should be adapted to the setting we're handling, i.e. medical and finance data are different. Always ask when you're unsure about something or you notice a significant difference between different settings, rather than simply making the decisions yourself.
