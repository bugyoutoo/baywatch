import pandas as pd
import matplotlib.pyplot as plt

# Read the metrics CSV file
metrics_df = pd.read_csv('training_metrics.csv')

# Create a figure
plt.figure(figsize=(10, 6))
plt.title('Training Loss Over Epochs')

# Plot Loss
plt.plot(metrics_df['Epoch'], metrics_df['Loss'], 'b-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('training_metrics.png')

# Show the plot
plt.show() 