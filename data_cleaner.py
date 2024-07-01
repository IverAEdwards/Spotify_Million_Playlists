import pandas as pd
import json
import os

directory = r"C:\Users\tup30579\Spotify Million\data_store\raw_data\spotify_million_playlist_dataset\data"
output_directory = r"C:\Users\tup30579\Spotify Million\data_store\named_data\cleaned_data_named" #for nameless, include 'names' in df.drop


def clean_track(track):
    track_uri = track.get('track_uri', '')
    track_id = track_uri.split('spotify:track:')[1]
    track_id = track_id.strip()
    track['track_id'] = track_id

    track.pop('track_uri', None)
    track.pop('pos', None)
    track.pop('artist_name', None)
    track.pop('artist_uri', None)
    track.pop('track_name', None)
    track.pop('album_uri', None)
    track.pop('album_name', None)
    track.pop('duration_ms', None)
    return track

def clean_json(data):
    playlists_raw = data["playlists"]
    df_playlists_raw = pd.DataFrame(playlists_raw)
    df_playlists = df_playlists_raw.drop(['collaborative', 'modified_at', 'num_tracks',
       'num_albums', 'num_followers', 'num_edits', 'duration_ms', 'num_artists',
       'description'], axis=1)

    df_playlists['tracks'] = df_playlists['tracks'].apply(lambda tracks: [clean_track(track) for track in tracks])
    playlists = df_playlists.to_dict(orient='records')

    return playlists

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
        
        # Open and load the JSON file
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    cleaned_data = clean_json(data)

    output_filename = f"cleaned_{filename}"
    output_path = os.path.join(output_directory, output_filename)
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(cleaned_data, json_file, ensure_ascii=False, indent=4)

print("cleaned JSON files have been saved to the new folder.")