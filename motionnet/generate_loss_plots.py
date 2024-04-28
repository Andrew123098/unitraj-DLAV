import pandas as pd
import matplotlib.pyplot as plt

# Specify the path to your .csv file
csv_path = "lightning_logs/version_39/metrics.csv"

# Read the .csv file
df = pd.read_csv(csv_path)

# Extract the odd-numbered rows starting from row 3
df = df.iloc[3::2]

# Extract the epochs and loss columns
epochs = df.iloc[:, 0]
loss = df.iloc[:, 3]

# Plot the loss curve
plt.plot(epochs, loss)
plt.xlabel("Epochs")
plt.ylabel("MinADE6")

# Save the plot as a .png file
plt.savefig("plots/cos_100epochs.png")