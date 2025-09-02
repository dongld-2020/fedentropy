import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV files
try:
    #fedavg_data = pd.read_csv("outcomes/metrics_lenet5_fedavg_dong_clients50_rounds100_clientsperround5.csv")
    #fednolowe_data = pd.read_csv("outcomes/metrics_cnn_fednolowe_dong_clients50_rounds100_clientsperround10.csv")
    #fedprox_data = pd.read_csv("outcomes/metrics_cnn_fedprox_clients50_rounds100_clientsperround15.csv")
    #fednova_data = pd.read_csv("outcomes/metrics_cnn_fednova_clients50_rounds100_clientsperround15.csv")
    #fedma_data = pd.read_csv("outcomes/metrics_cnn_fedma_clients50_rounds100_clientsperround15.csv")
    fedasl_data = pd.read_csv("outcomes/metrics_lenet5_fedasl_dong_clients50_rounds50_clientsperround5.csv")
    #fedlaw_data = pd.read_csv("outcomes/metrics_cnn_fedlaw_clients50_rounds100_clientsperround15.csv")    
    fedentropy_data = pd.read_csv("outcomes/metrics_lenet5_fedentropy_dong_clients50_rounds50_clientsperround5.csv")    
    fedmoon_data = pd.read_csv("outcomes/metrics_lenet5_moon_dong_clients50_rounds50_clientsperround5.csv")    
    
except FileNotFoundError:
    print("File not found. Please check the file paths.")
    exit()

# Ensure necessary columns exist
#required_columns = {"Round", "Global Validation Loss", "Global Train Loss", "Global Top1 Accuracy (%)"}
#if not required_columns.issubset(fedentropy_data.columns):
#    print("Missing required columns in the dataset. Check CSV file.")
#    exit()

# Convert round column to integer
fedentropy_data["Round"] = fedentropy_data["Round"].astype(int)

# Get min and max rounds
min_round = fedentropy_data["Round"].min()
max_round = fedentropy_data["Round"].max()

# Define plot settings
fig, axes = plt.subplots(1, 3, figsize=(32, 7), sharex=True)

# List of models with special styling for FedNoLoWe
models = [
    ("FedEntropy (our)", fedentropy_data, "-", 3),  # Solid & thicker
    #("FedAvg", fedavg_data, "--", 2),  # Dashed
    #("FedProx", fedprox_data, "--", 3),
    #("FedNova", fednova_data, "--", 2),
    #("FedMa", fedma_data, "--", 3),
    ("FedAsl", fedasl_data, "--", 2),
    #("FedLaw", fedlaw_data, "--", 3),
    #("FedNolowe", fednolowe_data, "--", 3),
     ("Moon", fedmoon_data, "--", 2)
    ]

# --- Plot Training Loss ---
for label, data, linestyle, linewidth in models:
    axes[0].plot(data["Round"], data["Global Train Loss"], label=label, linestyle=linestyle, linewidth=linewidth)
axes[0].set_xlabel("Communication Rounds", fontsize=22)
axes[0].set_ylabel("Training Loss", fontsize=22)
axes[0].tick_params(axis="x", labelsize=18)
axes[0].tick_params(axis="y", labelsize=18)
axes[0].grid(True, linestyle="--", alpha=0.7)
axes[0].set_title("(a) Training Loss", fontsize=22)

# --- Plot Validation Loss ---
for label, data, linestyle, linewidth in models:
    axes[1].plot(data["Round"], data["Global Validation Loss"], label=label, linestyle=linestyle, linewidth=linewidth)
axes[1].set_xlabel("Communication Rounds", fontsize=22)
axes[1].set_ylabel("Validation Loss", fontsize=22)
axes[1].tick_params(axis="x", labelsize=18)
axes[1].tick_params(axis="y", labelsize=18)
axes[1].grid(True, linestyle="--", alpha=0.7)
axes[1].set_title("(b) Validation Loss", fontsize=22)

# --- Plot Accuracy ---
for label, data, linestyle, linewidth in models:
    axes[2].plot(data["Round"], data["Global Top1 Accuracy (%)"], label=label, linestyle=linestyle, linewidth=linewidth)
axes[2].set_xlabel("Communication Rounds", fontsize=22)
axes[2].set_ylabel("Accuracy (%)", fontsize=22)
axes[2].tick_params(axis="x", labelsize=18)
axes[2].tick_params(axis="y", labelsize=18)
axes[2].grid(True, linestyle="--", alpha=0.7)
axes[2].set_title("(c) Accuracy", fontsize=22)

# Adjust x-ticks range
for ax in axes:
    ax.set_xticks(range(0, max_round + 10, 10))

# Adjust layout: Increase bottom padding so legend is fully visible
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.25, wspace=0.25)

# Place legend in 2 rows (4 items on top, 3 on bottom), now fully visible
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.001), fontsize=20, ncol=6, columnspacing=1.2, frameon=False)

# Show plots
plt.show()
