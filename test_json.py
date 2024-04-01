import json
import logging
import re

# Path to the JSON file
json_file_path = 'data-11222688005855917365.json'

# Read and parse the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

print(f"data {data}")
# Extract the Merge_point_addr value
merge_point_addr = data.get('Merge_point_addr', None)

print(f'Merge_point_addr: {merge_point_addr}')
