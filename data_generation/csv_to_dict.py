import collections
from json import dump
from pathlib import Path

import pandas as pd


def combine_dicts(dicts):
    combined_dict = collections.defaultdict(list)
    for d in dicts:
        for k, v in d.items():  # d.items() in Python 3+
            combined_dict[k].append(v)
    return combined_dict


emn_p = Path("data/emnist_data_inverted_labels.csv")
hasy_p = Path("data/hasy_data_resized_labels.csv")
unipen_p = Path("data/unipen_data_resized_refined.csv")


df1 = pd.read_csv(emn_p)
df2 = pd.read_csv(hasy_p)
df3 = pd.read_csv(unipen_p)


""" This section is for putting each csv into a dict"""
d1 = {}
for i in df1["character"].unique():
    d1[i] = [df1["image"][j] for j in df1[df1["character"] == i].index]

d2 = {}
for i in df2["character"].unique():
    d2[i] = [df2["image"][j] for j in df2[df2["character"] == i].index]

d3 = {}
for i in df3["character"].unique():
    d3[i] = [df3["image"][j] for j in df3[df3["character"] == i].index]


""" Combine them together """
dicts = [d1, d2, d3]
combined_dict = combine_dicts(dicts)


# for key, value in combined.items():
#     flat_list = [item for sublist in value for item in sublist]
#     combined[key] = flat_list

""" For each value in the dict checks if list is nested then flattens """
for key, value in combined_dict.items():
    if any(isinstance(i, list) for i in value):
        flat_list = [item for sublist in value for item in sublist]
        combined_dict[key] = flat_list


# for key in combined.keys():
#     if not isinstance(combined[key], list):
#         print(f"{key}:{isinstance(combined[key], list)}")

with open("data/combined_dict.json", "w") as f:
    dump(combined_dict, f)

