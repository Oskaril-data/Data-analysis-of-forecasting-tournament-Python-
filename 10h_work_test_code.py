# %%
#Importing libraries
import pandas as pd
import numpy as np
from scipy import stats
import os

# --- INSTRUCTIONS FOR EVALUATOR ---
# Please change the value of DATA_DIR to the folder containing the csv files.
# Use "." if the files are in the same folder as this script.

DATA_DIR = "."

df_daily = pd.read_csv(os.path.join(DATA_DIR, "rct-a-daily-forecasts.csv"))
df_qna = pd.read_csv(os.path.join(DATA_DIR, "rct-a-questions-answers.csv"))
df_PredictionSets = pd.read_csv(os.path.join(DATA_DIR, "rct-a-prediction-sets.csv"))

# %%
#STEP 2

# Create a working copy of the dataframe
work_df = df_PredictionSets.copy()

# ==========================================
# 1. DATA PREPROCESSING
# ==========================================

# Filter out forecasts that were made after the correct answer was already known.
work_df = work_df[work_df['made after correctness known'] != True]

# Convert the timestamp string to datetime objects for easier manipulation
work_df['timestamp'] = pd.to_datetime(work_df['prediction set created at'])

# Create a 'date' column to allow for daily grouping later
work_df['date'] = work_df['timestamp'].dt.date

# Sort the data to ensure correct selection of the latest forecast.
# Order: Question -> Answer -> User -> Date -> Time (ascending)
work_df = work_df.sort_values(
    by=['question id', 'answer id', 'membership guid', 'date', 'timestamp']
)

# Keep only the MOST RECENT forecast per user per day.
# If a user updated their forecast multiple times in one day, we only use the last one.
most_recent_forecasts = work_df.drop_duplicates(
    subset=['question id', 'answer id', 'membership guid', 'date'], 
    keep='last'
).copy()

# ==========================================
# 2. AGGREGATION HELPER FUNCTIONS
# ==========================================

def geometric_mean(x):
    # Method 3: Geometric Mean
    # Geometric mean handles zeros poorly (log(0) is undefined).
    # Clipping values to a safe range (0.001 - 0.999).
    x_clipped = np.clip(x, 0.001, 0.999)
    return stats.gmean(x_clipped)

def trimmed_mean_10(x):
    # Method 4: Trimmed Mean
    # Trimming the top and bottom 10% of forecasts to remove outliers.
    return stats.trim_mean(x, 0.1)

def geo_mean_odds(x):
    # Method 5: Geometric Mean of Odds
    # 1. Clip values to avoid division by zero in odds calculation
    p = np.clip(x, 0.001, 0.999)
    # 2. Convert probability to odds: p / (1 - p)
    odds = p / (1 - p)
    # 3. Calculate the geometric mean of the odds
    geo_odds = stats.gmean(odds)
    # 4. Convert back to probability: odds / (1 + odds)
    return geo_odds / (1 + geo_odds)


# ==========================================
# 3. AGGREGATION
# ==========================================


group_cols = ['discover question id', 'question id', 'answer id', 'date']


# Define aggregation rules.
# Applying statistical methods to 'forecasted probability'.
agg_rules = {
    'forecasted probability': [
        ('Raw Mean', 'mean'),
        ('Median', 'median'),
        ('Geometric Mean', geometric_mean),
        ('Trimmed Mean', trimmed_mean_10),
        ('Geo Mean Odds', geo_mean_odds),
        ('Count', 'count')
    ],
# For 'answer resolved probability', we use 'max'.
    # Since the probability is identical for all rows in the group, 
    # 'max' simply retrieves that constant value.
    'answer resolved probability': [('answer resolved probability', 'max')]
}

# Perform the grouping and aggregation
grouped = most_recent_forecasts.groupby(group_cols)
aggregated_results = grouped.agg(agg_rules)

# ==========================================
# 4. DATA CLEANUP
# ==========================================

# Flatten the MultiIndex columns created by the aggregation.
# This simplifies column names (e.g., changing ('forecasted probability', 'Raw Mean') to just 'Raw Mean').
aggregated_results.columns = [
    col[1] if col[0] == 'forecasted probability' else col[0] 
    for col in aggregated_results.columns
]

# Reset index to turn grouping keys back into regular columns
aggregated_results = aggregated_results.reset_index()


#Viewing the first 5 results
print(aggregated_results.head())


# %%

# STEP 3:
# ==========================================
# 5. ACCURACY CALCULATION (BRIER SCORE)
# ==========================================

# Creating a copy
df_scores = aggregated_results.copy()

# Ensure the 'answer resolved probability' column is numeric
df_scores['answer resolved probability'] = pd.to_numeric(df_scores['answer resolved probability'], errors='coerce')

# Filter: Remove rows where the question has not been resolved yet (NaN truth values)
df_scores = df_scores.dropna(subset=['answer resolved probability'])

# Calculate Squared Errors for each aggregation method
methods = ['Raw Mean', 'Median', 'Geometric Mean', 'Trimmed Mean', 'Geo Mean Odds']
score_cols = []

for method in methods:
    col_name = f'{method} SqError'
    # Brier Score Component: (Forecast - Outcome)^2
    df_scores[col_name] = (df_scores[method] - df_scores['answer resolved probability']) ** 2
    score_cols.append(col_name)


# Sum the Squared Errors at the Question-Date level.
brier_scores = df_scores.groupby(['question id', 'date'])[score_cols].sum()

# Calculate the final Mean Brier Score across the entire dataset
final_results = brier_scores.mean().sort_values()

# ==========================================
# 6. RESULTS OUTPUT
# ==========================================

print("=== Step 3: Aggregate Method Accuracy (Mean Brier Score) ===")
print()
print(pd.DataFrame(final_results, columns=['Mean Brier Score']))

# %%


# ==========================================
# STEP 4: IMPROVING THE METHOD (CALIBRATION / DAMPENING)
# ==========================================

def adjust_confidence(p, k=0.95):
    """
    Adjusts forecast confidence using the Karmarkar Equation.
    
    Parameters:
    - k > 1: Extremizes (pushes probs towards 0 or 1). Used if forecasts are underconfident.
    - k < 1: Dampens (pushes probs towards 0.5). Used if forecasts are overconfident.
    """
    # Clip values to prevent division by zero or log errors
    p = np.clip(p, 0.001, 0.999) 
    
    # Karmarkar Equation
    numerator = p ** k
    denominator = (p ** k) + ((1 - p) ** k)
    return numerator / denominator

# 1. Select the best method from Step 3
# 'Trimmed Mean' was the most accurate, but our analysis shows it might be slightly overconfident.
best_method = 'Trimmed Mean'

# 2. Apply the adjustment
# We observed that a k-factor of 0.95 improves the Brier Score.
# This implies the original Trimmed Mean was slightly "overconfident" (too close to 0 or 1).
# k=0.95 "dampens" the probabilities slightly towards 0.5 to improve calibration.
k_factor = 0.95
aggregated_results['Adjusted Forecast'] = adjust_confidence(aggregated_results[best_method], k=k_factor)

# ==========================================
# EVALUATING THE NEW METHOD
# ==========================================

# Create a working copy for scoring
df_scores_step4 = aggregated_results.copy()

# Ensure the resolution column is numeric and remove unresolved questions
df_scores_step4['answer resolved probability'] = pd.to_numeric(df_scores_step4['answer resolved probability'], errors='coerce')
df_scores_step4 = df_scores_step4.dropna(subset=['answer resolved probability'])

# Calculate Squared Error for the new method
df_scores_step4['Adjusted SqError'] = (df_scores_step4['Adjusted Forecast'] - df_scores_step4['answer resolved probability']) ** 2

# Recalculate the base method error for direct comparison
df_scores_step4['Base Method SqError'] = (df_scores_step4[best_method] - df_scores_step4['answer resolved probability']) ** 2

# Group by question and date
brier_scores_step4 = df_scores_step4.groupby(['question id', 'date'])[['Adjusted SqError', 'Base Method SqError']].sum()

# Calculate the final Mean Brier Score
mean_scores = brier_scores_step4.mean()

# Output the results
print(f"=== Step 4 Results: Calibration (Dampening) vs {best_method} ===")
print(f"Calibration Factor (k): {k_factor}")
print("-" * 40)
print(f"Original ({best_method}) Brier Score: {mean_scores['Base Method SqError']:.6f}")
print(f"Adjusted Forecast Brier Score:        {mean_scores['Adjusted SqError']:.6f}")
print("-" * 40)

# Check for improvement
improvement = mean_scores['Base Method SqError'] - mean_scores['Adjusted SqError']
if improvement > 0:
    print(f"SUCCESS: Dampening (k={k_factor}) improved the score by {improvement:.6f}!")
else:
    print("NOTE: The adjustment did not improve the score.")


