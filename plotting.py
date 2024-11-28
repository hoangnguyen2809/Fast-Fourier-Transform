import matplotlib.pyplot as plt
import pandas as pd
import sys

# Check if the user provided a filename
if len(sys.argv) != 2:
    print("Usage: python3 plotting.py <filename>")
    sys.exit(1)

# Get the filename from the command-line arguments
filename = sys.argv[1]

# Load data from the specified file
try:
    data = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

# Plot magnitude spectrum
plt.figure(figsize=(10, 6))
plt.plot(data["Frequency Bin"], data["Magnitude"], label="Magnitude")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")
plt.title("FFT Magnitude Spectrum")
plt.grid()
plt.legend()
plt.show()
