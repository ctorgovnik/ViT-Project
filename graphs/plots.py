import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 16  # Increase base font size

script_dir = os.path.dirname(os.path.abspath(__file__))

sns.set_theme(style="whitegrid")

data = {
    "pretrain_dataset": ["ImageNet-100", "ImageNet-1000"] * 2,
    "accuracy": [0.75, 0.55, 0.86, 0.80],
    "model": ["ViT-L", "ViT-L", "ResNet-18", "ResNet-18"],
}

df = pd.DataFrame(data)

df["x_pos"] = df["pretrain_dataset"].map(
    {
        "ImageNet-100": 1 / 4,
        "ImageNet-1000": 3 / 4,
    }
)

plt.figure(figsize=(8, 6))

sns.lineplot(
    x="x_pos",
    y="accuracy",
    hue="model",
    data=df,
    marker="o",
    linestyle="--",
    markersize=10,
    linewidth=2.5,
)

# Labels and ticks
plt.xlabel("Pretrain Dataset", fontsize=18)
plt.ylabel("Accuracy", fontsize=18)
plt.title("Accuracy vs Pretrain Dataset for Different Models", fontsize=20)
plt.xticks(ticks=[1 / 4, 3 / 4], labels=["ImageNet-100", "ImageNet-1000"], fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.legend(title="Model", fontsize=14, title_fontsize=16, loc="lower right")

# Adjust space
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

# Save and show
plt.tight_layout()
plot_path = os.path.join(script_dir, "my_plot.png")
plt.savefig(plot_path)
plt.show()
