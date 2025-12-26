Crowd Forecasting Accuracy & Calibration Analysis

***********THE DATASETS USED ARE NO LONGER AVAILABLE************************

Project Overview

This project analyzes "Wisdom of the Crowd" data to determine the most accurate method for aggregating individual probability forecasts. Using a dataset of daily probability predictions,
the script cleans the data, tests various statistical aggregation methods, calculates accuracy using Brier Scores, 
and further optimizes the best model using probability calibration techniques.

Key Objectives

    Data Preprocessing: Clean raw forecasting logs to isolate the most relevant "latest" prediction per user per day.

    Aggregation Strategy: Compare five distinct statistical methods to combine individual forecasts into a single "crowd consensus."

    Performance Evaluation: Use the Brier Score (Mean Squared Error for probabilities) to objectively rank the aggregation methods.

    Calibration Optimization: Apply the Karmarkar Equation to adjust confidence levels (dampening/extremizing) to minimize error.

Methodology
1. Data Cleaning

    Temporal Filtering: Removed forecasts made after the ground truth was known.

    Deduplication: Implemented logic to retain only the most recent forecast for a specific user on a specific day to prevent data redundancy.

    Structure: Aligned Question, Answer, and User data for time-series analysis.

2. Aggregation Algorithms

The script calculates the consensus probability using the following methods:

    Raw Mean: Standard arithmetic average.

    Median: Robust against outliers.

    Geometric Mean: Useful for probabilities but sensitive to zeros (handled via clipping).

    Trimmed Mean (10%): Removes the top/bottom 10% of forecasts to eliminate extreme outliers/trolls.

    Geometric Mean of Odds: Converts probabilities to odds space before averaging, often theoretically superior for probability aggregation.

3. Metric: The Brier Score

Accuracy is measured using the Brier Score. Lower scores indicate better accuracy.



4. Calibration (The Karmarkar Equation)

After identifying the Trimmed Mean as the best baseline method, I applied a calibration transformation to further improve the score.

    Dampening (k < 1): Pushes probabilities closer to 0.5 (correcting for overconfidence).

    Extremizing (k > 1): Pushes probabilities closer to 0 or 1 (correcting for underconfidence).

    Result: A factor of k=0.95 (slight dampening) yielded the lowest error, suggesting the crowd was slightly overconfident.

Technologies Used

    Python 3.13.5

    Pandas: For complex dataframe manipulation, grouping, and time-series handling.

    NumPy: For vectorized mathematical operations (clipping, logarithms).

    SciPy: For statistical functions (Geometric Mean, Trimmed Mean).

Project Structure

├── 10h_work_test_code.py   # Main analysis script
├── rct-a-daily-forecasts.csv      # (Input) Daily forecast logs
├── rct-a-questions-answers.csv    # (Input) Question metadata
├── rct-a-prediction-sets.csv      # (Input) User prediction history
└── README.md



Results Summary

The analysis determined that the Trimmed Mean was the most accurate standard aggregation method. However, by applying a calibration factor of 0.95 (dampening), 
the Mean Brier Score was further reduced, proving that a slightly conservative adjustment to the crowd's consensus yields better predictive power.


