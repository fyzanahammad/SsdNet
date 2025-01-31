import csv

# Generate model IDs with full paths (e.g., "data/FNSet/0_1.binvox")
filtered_models = []
for class_id in range(24):
    for model_num in range(1, 51):
        model_id = f"data/FNSet/{class_id}_{model_num}.binvox"  # Match minfo.csv format
        filtered_models.append(model_id)

# Filter minfo.csv to include only the desired models
with open('data/minfo.csv', 'r') as infile, open('data/minfo_test.csv', 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    for row in reader:
        if row[0] in filtered_models:  # Check full path (e.g., "data/FNSet/0_1.binvox")
            writer.writerow(row)