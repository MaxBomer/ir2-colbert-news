import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("colbert_variants_auc.csv")

x = df["Step"].astype(int)

series = {
    "ColBERT (base)": "ColBERT-2025-12-10_00-28-49 - val/auc",
    "ColBERT + user_attention": "ColBERTAttn-2025-12-10_00-28-49 - val/auc",
    "ColBERT + position_embeddings": "ColBERTPos-2025-12-10_00-28-50 - val/auc",
    "ColBERT + hierarchical_attention": "ColBERTHier-2025-12-10_00-28-48 - val/auc",
}

plt.figure(figsize=(7, 4))
for label, col in series.items():
    if col in df.columns:
        plt.plot(x, df[col], label=label)

plt.xlabel("Training step (validation checkpoints)")
plt.ylabel("Validation AUC")
plt.title("Training dynamics of ColBERT variants (NRMS-style)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("colbert_variants_auc.png", dpi=200)
plt.show()