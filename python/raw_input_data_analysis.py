import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/mnt/data/20260812-sum-700(1).csv"
df = pd.read_csv(csv_path, encoding="gb18030")

df["Norm_before"] = pd.to_numeric(df["Norm_before"], errors="coerce")
df["Norm_after"] = pd.to_numeric(df["Norm_after"], errors="coerce")

mask = (df["Norm_before"] > 5) & (df["Norm_after"] > 5)
high = df.loc[mask, ["ID", "Norm_before", "Norm_after"]].copy()

plt.figure(figsize=(8, 7))
plt.scatter(df["Norm_before"], df["Norm_after"], alpha=0.55, label="All samples")
plt.scatter(
    high["Norm_before"],
    high["Norm_after"],
    alpha=0.9,
    label="Norm_before > 5 and Norm_after > 5"
)
plt.axvline(5, linestyle="--", linewidth=1)
plt.axhline(5, linestyle="--", linewidth=1)
plt.xlabel("Norm_before")
plt.ylabel("Norm_after")
plt.title("Norm_before vs Norm_after")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

out = "/mnt/data/norm_before_vs_after.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.show()

print(f"两者均 > 5 的样本数: {len(high)}")
print(high.to_string(index=False))
print(f"\n图已保存到: {out}")
