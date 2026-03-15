import matplotlib.pyplot as plt
import os

os.makedirs("charts", exist_ok=True)

opt_levels = ['-O0', '-O3']
# 55.89 Billion vs 17.92 Billion
instructions = [55893216490, 17929693264] 

plt.figure(figsize=(7, 5))
plt.bar(opt_levels, instructions, color=['red', 'blue'])
plt.title('Executed Instructions vs Compiler Optimization')
plt.ylabel('Total Instructions (Billions)')
plt.xlabel('Optimization Flag')

# Format the numbers to look clean (e.g., 55.9B)
for i, v in enumerate(instructions):
    plt.text(i, v, f"{v/1000000000:.2f}B", ha='center', va='bottom')

plt.savefig("charts/instruction_count.png")
print(" Generated charts/instruction_count.png")