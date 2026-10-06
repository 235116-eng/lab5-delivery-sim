import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

delivery_times = np.random.normal(32.4, 8.1, 100)
delivery_times[:3] = [68, 71, 75]
delivery_times = np.round(delivery_times, 1)

df = pd.DataFrame({
    "OrderID": range(1, 101),
    "DeliveryTime(min)": delivery_times
})

df.to_csv("delivery_times.csv", index=False)

mean = df["DeliveryTime(min)"].mean()
std = df["DeliveryTime(min)"].std()

print(f"Real delivery times -- Mean: {mean:.1f} min Std: {std:.1f} min")

simulated = np.random.normal(mean, std, 100)

print(f"Simulated 100 values from Normal(mean={mean:.1f}, std={std:.1f})")

missing = df.isnull().sum()

print("Missing values:", missing.sum())

q1 = df["DeliveryTime(min)"].quantile(0.25)
q2 = df["DeliveryTime(min)"].quantile(0.50)
q3 = df["DeliveryTime(min)"].quantile(0.75)

print(f"Percentiles -- 25th: {q1:.1f} 50th: {q2:.1f} 75th: {q3:.1f}")

iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[
    (df["DeliveryTime(min)"] < lower_bound) |
    (df["DeliveryTime(min)"] > upper_bound)
]

print(f"IQR outlier bounds: {lower_bound:.1f} - {upper_bound:.1f}")
print(f"Outliers detected: {len(outliers)} deliveries ({outliers['DeliveryTime(min)'].tolist()})")

percentage = (len(outliers) / len(df)) * 100

print(f"Percentage flagged as outliers: {percentage:.1f}%")

plt.figure(figsize=(10, 6))
plt.hist(
    df["DeliveryTime(min)"],
    bins=15,
    alpha=0.6,
    label="Real Delivery Times"
)
plt.hist(
    simulated,
    bins=15,
    alpha=0.6,
    label="Simulated Delivery Times"
)
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Frequency")
plt.title("Real vs Simulated Delivery Times")
plt.legend()
plt.tight_layout()
plt.savefig("delivery_comparison_histogram.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.boxplot(df["DeliveryTime(min)"])
plt.ylabel("Delivery Time (minutes)")
plt.title("Delivery Time Boxplot")
plt.tight_layout()
plt.savefig("delivery_boxplot.png")
plt.close()

print("Saved delivery_comparison_histogram.png, delivery_boxplot.png")