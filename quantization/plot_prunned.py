import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = "prune_results.csv"
df = pd.read_csv(csv_path)
df_sorted = df.sort_values("acc after 90% prune", ascending=False)

plt.figure(figsize=(14, 6))
sns.barplot(data=df_sorted, x="layer", y="acc after 90% prune", palette="viridis")
plt.xticks(rotation=90)
plt.title("Accuracy After Pruning 90% of Each Layer (One Layer at a Time)")
plt.xlabel("Layer")
plt.ylabel("Accuracy (%)")
plt.tight_layout()

plot_path = "prune_accuracy_plot.png"
plt.savefig(plot_path)
df_sorted.tail(5), df_sorted.head(5)
