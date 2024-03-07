import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Replace 'file_path.csv' with the path to your CSV file
df = pd.read_csv('weight-signs-training/data/mnist_no_conv.csv')
val_accuracy_df = df[df['metric'] == 'val_accuracy']
average_val_accuracy = val_accuracy_df.groupby('epoch')['value'].mean().reset_index()['value'].tolist()
runs_val_accuracy_history = val_accuracy_df.groupby('run')['value'].apply(list).reset_index()['value'].tolist()


# Replace 'file_path.csv' with the path to your CSV file
df_comp = pd.read_csv('weight-signs-training/data/data_multiplier/mnist_no_conv_num7_1.csv')
val_accuracy_df_comp = df_comp[df_comp['metric'] == 'val_accuracy']
average_val_accuracy_comp = val_accuracy_df_comp.groupby('epoch')['value'].mean().reset_index()['value'].tolist()
runs_val_accuracy_history_comp = val_accuracy_df_comp.groupby('run')['value'].apply(list).reset_index()['value'].tolist()

# Model 1 data
data_model_1 = runs_val_accuracy_history
average_data_model_1 = average_val_accuracy

# Model 2 data (with a slight modification to differentiate)
data_model_2 = runs_val_accuracy_history_comp
average_data_model_2 = average_val_accuracy_comp

# Plotting
plt.figure(figsize=(10, 6))

# Plot each run of Model 1 with lighter color
for i in range(100):
    plt.plot(data_model_1[i][:], color='blue', alpha=0.05)

# Plot average of Model 1 with darker color
# Plot each run of Model 2 with lighter color
for i in range(100):
    plt.plot(data_model_2[i][:], color='red', alpha=0.05)

# Plot average of Model 2 with darker color
plt.plot(average_data_model_1, color='blue', label='Average Accuracy Model 1', linewidth=2)

plt.plot(average_data_model_2, color='red', label='Average Accuracy Model 2', linewidth=2)

plt.title('Average History of Accuracy over 100 Runs for Two Different Models')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("weight-signs-training/data_multiplier")
