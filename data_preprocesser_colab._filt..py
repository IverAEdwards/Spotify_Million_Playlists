from scipy.sparse import csr_matrix, save_npz
import pandas as pd
import json
from implicit.evaluation import leave_k_out_split


file_path = r"C:\Users\tup30579\Spotify Million\data_store\super_data.json"

train_csr_path = r"C:\Users\tup30579\Spotify Million\data_store\train_csr.npz"
test_csr_path = r"C:\Users\tup30579\Spotify Million\data_store\test_csr.npz"


output_mapping_path = r"C:\Users\tup30579\Spotify Million\data_store\track_id_to_int.json"
output_reverse_mapping_path = r"C:\Users\tup30579\Spotify Million\data_store\int_to_track_id.json"

with open(file_path, 'r', encoding='utf-8') as file:
    playlists = json.load(file)

track_id_to_int = {}
int_to_track_id = {}
user_item_interactions = []

for idx, playlist in enumerate(playlists):
    for track in playlist['tracks']:
        track_id = track['track_id']
        if track_id not in track_id_to_int:
            int_id = len(track_id_to_int)
            track_id_to_int[track_id] = int_id
            int_to_track_id[int_id] = track_id
        user_item_interactions.append((idx, track_id_to_int[track_id]))

interactions_df = pd.DataFrame(user_item_interactions, columns=['user_id', 'track_id'])
interactions_df['presence'] = 1

sparse_user_item_matrix = csr_matrix((interactions_df['presence'], 
                                     (interactions_df['user_id'], interactions_df['track_id'])))


train_csr,test_csr = leave_k_out_split(sparse_user_item_matrix, K=10, random_state=None)

save_npz(train_csr_path, train_csr)
save_npz(test_csr_path, test_csr)

# Save the mappings to JSON files
with open(output_mapping_path, 'w', encoding='utf-8') as f:
    json.dump(track_id_to_int, f)

with open(output_reverse_mapping_path, 'w', encoding='utf-8') as f:
    json.dump(int_to_track_id, f)