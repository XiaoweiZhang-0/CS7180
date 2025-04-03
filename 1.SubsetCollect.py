import json

with open('first_200000_entries.json') as f:
    data = json.load(f)

# Take first 1000 entries (or adjust the number as needed)
subset = data[:200000]

# Save to a new file
with open('subset.json', 'w') as out_file:
    json.dump(subset, out_file, indent=2)  # indent=2 makes it pretty
