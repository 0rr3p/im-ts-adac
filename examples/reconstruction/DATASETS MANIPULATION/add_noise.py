import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('QUERY_CSV_SANE2_EXCEL.csv', sep=';', decimal=',')

# Inspect j5 columns
j5_cols = ['j5_v', 'j5_a', 'j5_t']
print(df[j5_cols].describe())

# Add slight random noise (e.g., 7% of standard deviation)
np.random.seed(42)
for col in j5_cols:
    std_dev = df[col].std()
    noise = np.random.normal(0, 0.07 * std_dev, size=len(df))
    df[col] = df[col] + noise

# Save the modified dataframe
output_filename = 'QUERY_CSV_SANE2_EXCEL_noisy_j5.csv'
df.to_csv(output_filename, sep=';', decimal=',', index=False)
print(f"Saved to {output_filename}")