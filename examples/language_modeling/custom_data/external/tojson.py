import json

input_file_path = "data.txt"
output_file_path = "data.json"

data = []

# Read the text file line by line
with open(input_file_path, "r", encoding="utf-8") as file:
    for line in file:
        data.append({"text": line.strip()})

with open(output_file_path, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=2)

print("Conversion completed. The JSON file has been saved.")
