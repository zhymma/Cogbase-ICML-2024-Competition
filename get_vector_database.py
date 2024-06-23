from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import torch
import numpy as np
import json

training_data = r"1-1-Formalization\new_training_data\training_data.json"
traing_label = r"1-1-Formalization\new_training_data\training_label.json"
with open(training_data, 'r', encoding="utf8") as f:
    training_data = json.load(f)
with open(traing_label, 'r', encoding="utf8") as f:
    training_label = json.load(f)
informal_statements = [i["informal_statement"] for i in training_data]
informal_proofs = [i["informal_proof"] for i in training_data]
with torch.no_grad():
    model = BGEM3FlagModel('./bge-m3',  use_fp16=True,device='cuda',pooling_method='mean')
    informal_statements_embeddings = model.encode(informal_statements, 
                                    batch_size=12, 
                                    max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                    )['dense_vecs']
    informal_statements_embedding = torch.from_numpy(np.array(informal_statements_embeddings))
    informal_proofs_embeddings = model.encode(informal_proofs,
                                    batch_size=12, 
                                    max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                    )['dense_vecs']
    informal_proofs_embedding = torch.from_numpy(np.array(informal_proofs_embeddings))
    vector_library = {
        "informal_statements_embedding": informal_statements_embedding,
        "informal_proofs_embedding": informal_proofs_embedding,
    }
    torch.save(vector_library, "1-1-Formalization/new_training_data/" + f"vector_library.pt")
    del vector_library

# load the vector library
vector_library = torch.load("1-1-Formalization/new_training_data/" + f"vector_library.pt")

temp = ["Jenny is planning to make a rectangular garden plot in her yard. She wants the length of the plot to be 12 feet and the width to be 8 feet. She marks the four corners of the plot on a coordinate plane as follows: (0,0), (12,0), (0,8), and (12,8). \n\n1. What are the coordinates of the center of the plot? \n2. How many square feet is the garden plot?"]
temp1 = ["1. The center of the plot is the average of the x-coordinates and the y-coordinates. So, the x-coordinate of the center is (0+12)/2 = 6 and the y-coordinate of the center is (0+8)/2 = 4. Therefore, the center of the plot is at (6,4).\n\n2. The area of a rectangle is calculated by multiplying the length by the width. Here, the length is 12 feet and the width is 8 feet, so the area is 12*8 = 96 square feet. Therefore, the garden plot is 96 square feet."]
with torch.no_grad():
    embeddings1 = model.encode(temp,
                                    batch_size=12, 
                                    max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                    )['dense_vecs'] 
    embeddings2 = model.encode(temp1,

                                    batch_size=12,
                                    max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                    )['dense_vecs']
    embeddings1 = torch.from_numpy(np.array(embeddings1)).to('cuda')
    embeddings2 = torch.from_numpy(np.array(embeddings2)).to('cuda')

    informal_statements_embedding = vector_library["informal_statements_embedding"].to('cuda')
    informal_proofs_embedding = vector_library["informal_proofs_embedding"].to('cuda')
    # calculate the cosine similarity
    sim = embeddings1 @ informal_statements_embedding.T
    k = 5
    seleted_idxs1 = torch.topk(sim, k=k).indices.squeeze(0).tolist()
    print(seleted_idxs1)

    sim = embeddings2 @ informal_proofs_embedding.T
    k = 5
    seleted_idxs2 = torch.topk(sim, k=k).indices.squeeze(0).tolist()
    print(seleted_idxs2)

    all_selected_idxs = list(set(seleted_idxs1 + seleted_idxs2))
    print(all_selected_idxs)

