import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('fitnesses_weighteddeepset.csv')

# Assume the first column is 'time' or similar, or use index as time
time = df.iloc[:, 0]
columns_to_plot = df.columns[1:4]  # First 3 columns after time
custom_labels = ['sigma', 'average', 'maximum']

plt.figure(figsize=(10, 6))

# First plot as line, next two as scatter
plt.plot(time, df[columns_to_plot[0]], label=custom_labels[0])
plt.scatter(time, df[columns_to_plot[1]], label=custom_labels[1])
plt.scatter(time, df[columns_to_plot[2]], label=custom_labels[2])

plt.xlabel(df.columns[0])
plt.ylabel('Value')
plt.title('First 3 Columns Over Time')
plt.legend()
plt.grid(True)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()
