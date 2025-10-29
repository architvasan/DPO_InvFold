import pandas as pd
import json

def get_prefix(filename):
    return filename.rsplit('_', 1)[0]

data_loc = '/flare/FoundEpidem/avasan/IDEAL/PeptideDesign/NMNAT_2_Screens/Trial_FromExperiment/runs'
df_good = pd.read_csv(f'{data_loc}/step4/developable.csv')
df_all = pd.read_csv(f'{data_loc}/step3/energies.csv')

df_bad = df_all[~df_all['seqs'].isin(df_good['seqs'])]

print(df_bad)

dataset_test = []
for it, row in df_good.iterrows():
    if it>1000:
        break
    pref_seq = row['seqs']
    pdb_it = row['pdbs']
    pdb_match = get_prefix(pdb_it)
    matches = df_bad[df_bad['pdbs'].str.contains(pdb_match, na=False)]
    if len(matches)>=1:
        print(pdb_match)
        #print(list(matches['seqs'])[0])
        unpref_seq = list(matches['seqs'])[0]
        unpref_seq = str(unpref_seq)
        if unpref_seq == 'nan':
            continue
        print(unpref_seq)
        dataset_test.append({'pdb_file': f'{data_loc}/step3/{pdb_it}', 
                             'preferred_seq': pref_seq,
                             'unpreferred_seq': unpref_seq 
                            })


# Specify the filename for the JSON output
json_filename = "test_data_real.json"

# Open the file in write mode ('w') and use json.dump() to write the list
try:
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(dataset_test, f, indent=4)
    print(f"List successfully written to {json_filename}")
except IOError as e:
    print(f"Error writing to file: {e}")
