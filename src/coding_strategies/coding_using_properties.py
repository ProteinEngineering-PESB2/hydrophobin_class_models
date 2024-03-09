#libraries
import pandas as pd
from lib.physicochemical_properties import physicochemical_encoder
from lib.fft_encoding import fft_encoding
import sys

#data to encode
input_data = pd.read_csv(sys.argv[1])
path_export = sys.argv[2]
column_with_id = "target"

dataset_encoder = pd.read_csv("/home/dmedina/clustering_protein_project/src/coding_strategies/clustering_encoders.csv")
dataset_encoder.index = dataset_encoder['residue']

#define variables
column_with_seq = "sequence"

for group in ["Group_0","Group_1","Group_2","Group_3","Group_4","Group_5","Group_6","Group_7"]:
    print("Processing: ", group)
    physicochemical_encoder_instance = physicochemical_encoder(
        input_data,
        group,
        dataset_encoder,
        column_with_seq,
        column_with_id
    )

    df_encoding = physicochemical_encoder_instance.encoding_dataset()
    df_encoding.to_csv(f"{path_export}{group}.csv", index=False)

    fft_instance = fft_encoding(
        df_encoding, 
        len(df_encoding.columns)-1, 
        column_with_id
    )

    df_data_fft = fft_instance.encoding_dataset()
    df_data_fft.to_csv(f"{path_export}{group}_FFT.csv", index=False)
