# ============================================
# Student Performance Prediction Project
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

plt.style.use("default")

# ============================================
# 1. Load Dataset
# ============================================

df = pd.read_excel("data/Student_Performance_ML_300_Rows.xlsx")
print("Dataset Loaded Successfully")

# ============================================
# 2. Data Preprocessing
# ============================================

df_ml = df.drop(columns=["Student_ID"])

encoder = LabelEncoder()
df_ml["Internet_Access"] = encoder.fit_transform(df_ml["Internet_Access"])
df_ml["Result"] = encoder.fit_transform(df_ml["Result"])  # Pass=1, Fail=0

# ============================================
# 3. EDA – 15 QUESTIONS WITH PROPER DATA LABELS
# ============================================

# Q1: Pass vs Fail Distribution (Bar + labels)
counts = df["Result"].value_counts()
ax = counts.plot(kind="bar", title="Pass vs Fail Distribution")
for i, v in enumerate(counts):
    ax.text(i, v + 2, str(v), ha='center')
plt.ylabel("Number of Students")
plt.show()

# Q2: Attendance vs Result (Boxplot + median labels)
ax = df.boxplot(column="Attendance", by="Result")
medians = df.groupby("Result")["Attendance"].median()
for i, median in enumerate(medians):
    ax.text(i + 1, median, f"Median: {median:.1f}", ha="center")
plt.title("Attendance vs Result")
plt.suptitle("")
plt.show()

# Q3: Study Hours vs Internal Marks (Scatter + correlation)
plt.scatter(df["Study_Hours"], df["Internal_Marks"])
corr = df["Study_Hours"].corr(df["Internal_Marks"])
plt.title(f"Study Hours vs Internal Marks (Corr = {corr:.2f})")
plt.xlabel("Study Hours")
plt.ylabel("Internal Marks")
plt.show()

# Q4: Internal Marks vs Result (Boxplot + median)
ax = df.boxplot(column="Internal_Marks", by="Result")
medians = df.groupby("Result")["Internal_Marks"].median()
for i, median in enumerate(medians):
    ax.text(i + 1, median, f"{median:.1f}", ha="center")
plt.title("Internal Marks vs Result")
plt.suptitle("")
plt.show()

# Q5: Test Score vs Assignment Score (Scatter + correlation)
plt.scatter(df["Test_Score"], df["Assignment_Score"])
corr = df["Test_Score"].corr(df["Assignment_Score"])
plt.title(f"Test vs Assignment Score (Corr = {corr:.2f})")
plt.xlabel("Test Score")
plt.ylabel("Assignment Score")
plt.show()

# Q6: Participation Score vs Result (Boxplot + median)
ax = df.boxplot(column="Participation_Score", by="Result")
medians = df.groupby("Result")["Participation_Score"].median()
for i, median in enumerate(medians):
    ax.text(i + 1, median, f"{median:.1f}", ha="center")
plt.title("Participation Score vs Result")
plt.suptitle("")
plt.show()

# Q7: Assignments Submitted vs Result (Boxplot + median)
ax = df.boxplot(column="Assignments_Submitted", by="Result")
medians = df.groupby("Result")["Assignments_Submitted"].median()
for i, median in enumerate(medians):
    ax.text(i + 1, median, f"{median:.0f}", ha="center")
plt.title("Assignments Submitted vs Result")
plt.suptitle("")
plt.show()

# Q8: Extra-Curricular Score vs Result (Boxplot + median)
ax = df.boxplot(column="Extra_Curricular_Score", by="Result")
medians = df.groupby("Result")["Extra_Curricular_Score"].median()
for i, median in enumerate(medians):
    ax.text(i + 1, median, f"{median:.1f}", ha="center")
plt.title("Extra-Curricular Score vs Result")
plt.suptitle("")
plt.show()

# Q9: Internet Access vs Result (Grouped bar + labels)
ax = df.groupby(["Internet_Access", "Result"]).size().unstack().plot(kind="bar")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Internet Access vs Result")
plt.ylabel("Number of Students")
plt.show()

# Q10: Study Environment vs Result (Boxplot + median)
ax = df.boxplot(column="Study_Environment_Score", by="Result")
medians = df.groupby("Result")["Study_Environment_Score"].median()
for i, median in enumerate(medians):
    ax.text(i + 1, median, f"{median:.1f}", ha="center")
plt.title("Study Environment vs Result")
plt.suptitle("")
plt.show()

# Q11: Backlogs vs Result (Grouped bar + labels)
ax = df.groupby(["Backlogs", "Result"]).size().unstack().plot(kind="bar")
for container in ax.containers:
    ax.bar_label(container)
plt.title("Backlogs vs Result")
plt.ylabel("Number of Students")
plt.show()

# Q12: Stress Level vs Result (Boxplot + median)
ax = df.boxplot(column="Stress_Level_Score", by="Result")
medians = df.groupby("Result")["Stress_Level_Score"].median()
for i, median in enumerate(medians):
    ax.text(i + 1, median, f"{median:.1f}", ha="center")
plt.title("Stress Level vs Result")
plt.suptitle("")
plt.show()

# Q13: Previous Performance vs Result (Boxplot + median)
ax = df.boxplot(column="Previous_Performance_Score", by="Result")
medians = df.groupby("Result")["Previous_Performance_Score"].median()
for i, median in enumerate(medians):
    ax.text(i + 1, median, f"{median:.1f}", ha="center")
plt.title("Previous Performance vs Result")
plt.suptitle("")
plt.show()

# Q14: Attendance Distribution (Histogram + total count)
plt.hist(df["Attendance"], bins=20)
plt.title("Attendance Distribution")
plt.xlabel("Attendance")
plt.ylabel("Frequency")
plt.text(0.7, 0.9, f"Total Students: {len(df)}", transform=plt.gca().transAxes)
plt.show()

# Q15: Study Hours Distribution (Histogram + mean)
plt.hist(df["Study_Hours"], bins=10)
mean_val = df["Study_Hours"].mean()
plt.axvline(mean_val, linestyle="--")
plt.title(f"Study Hours Distribution (Mean = {mean_val:.2f})")
plt.xlabel("Study Hours")
plt.ylabel("Frequency")
plt.show()

# ============================================
# 4. Machine Learning
# ============================================

X = df_ml.drop(columns=["Result"])
y = df_ml["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
ax = importance.plot(kind="barh", title="Feature Importance")
for i, v in enumerate(importance):
    ax.text(v, i, f"{v:.2f}", va="center")
plt.show()
