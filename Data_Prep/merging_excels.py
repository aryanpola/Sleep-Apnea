import pandas as pd

# Load files
labels_df = pd.read_csv(r"Data_Prep\Labels Three Events.csv")
distances_df = pd.read_csv(r"Data_Prep\Multiple Distances Three Events.csv")
distance_only = distances_df[["Dist1", "Dist2", "Dist3", "Dist4", "Dist5"]]

# Repeat labels: one per 170 rows
repeated_labels = labels_df.loc[labels_df.index.repeat(170)].reset_index(drop=True)
distance_only["Label"] = repeated_labels

# 🔥 Remove rows with label == 2
filtered_df = distance_only[distance_only["Label"] != 2].reset_index(drop=True)

# Save to Excel
output_path = r"C:\Users\aryan\OneDrive\Documents\sohams_dop\Data_Prep\CNN_Input_Labeled_0_1_only.xlsx"
filtered_df.to_excel(output_path, index=False)

print(f"✅ Filtered and saved successfully to: {output_path}")
