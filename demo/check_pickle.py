import pickle
import numpy as np

file_path = "/Users/fabeun/Documents/senior research/collected_dataset/BuildingB/stable/002.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

print("Type:", type(data))

# If it's a dict, print keys
if isinstance(data, dict):
    print("Keys:", data.keys())
    for k in data.keys():
        v = data[k]
        print(f"\nKey: {k}")
        print("  Type:", type(v))
        if hasattr(v, "shape"):
            print("  Shape:", v.shape)
        else:
            try:
                print("  Length:", len(v))
            except:
                pass
else:
    print("Data is not a dict. Content:", data)
