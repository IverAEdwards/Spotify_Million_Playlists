import os
import json
import ijson

cleaned_data_directory = r"C:\Users\tup30579\Spotify Million\data_store\named_data\cleaned_data_named"
output_file1 = r"C:\Users\tup30579\Spotify Million\data_store\named_data\all_names.json"
output_file2 = r"C:\Users\tup30579\Spotify Million\data_store\named_data\all_tracks.json"

all_names = []
all_tracks = []

# Iterate through each cleaned JSON file in the directory
for filename in os.listdir(cleaned_data_directory):
    filepath = os.path.join(cleaned_data_directory, filename)
    
    # Read the cleaned JSON file
    with open(filepath, 'r', encoding='utf-8') as file:

        objects = ijson.items(file, 'item')
        for obj in objects:
            names = obj['name']
            tracks = [track["track_id"] for track in obj["tracks"]]
            all_names.append(names)
            all_tracks.append(tracks)

# merge the data to a single JSON file
with open(output_file1, 'w', encoding='utf-8') as output_json_file:
    json.dump(all_names, output_json_file, ensure_ascii=False, indent=4)

with open(output_file2, 'w', encoding='utf-8') as output_json_file:
    json.dump(all_tracks, output_json_file, ensure_ascii=False, indent=4)
