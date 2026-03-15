import matplotlib.pyplot as plt
import os

os.makedirs("charts", exist_ok=True)
# Actual MSE values from your Softmax output logs
errors = [1.51e-09, 2.70e-11, 1.45e-17, 5.65e-14, 6.58e-10]
samples = ['Sample 0', 'Sample 1', 'Sample 2', 'Sample 3', 'Sample 4']

plt.figure(figsize=(7, 4))
plt.bar(samples, errors, color='teal')
plt.yscale('log')
plt.title('Softmax MSE Distribution Across Validation Samples')
plt.ylabel('Mean Squared Error (Log Scale)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("error_distribution.png")
print("Generated error_distribution.png")