import os

import pandas as pd

path = "./unipen_data_resized_refined/"
char_list = []
dir_list = []
for file in os.listdir(path):
    name = file.split("_")
    if name[1] != "unicode":
        char_list += [name[1]]
        dir_list += [f"data/unipen_data_resized_refined/{file}"]
    else:
        if name[2].isdigit() and name[3].isdigit():
            char_list += [f"un{name[3]}"]
            dir_list += [f"data/unipen_data_resized_refined/{file}"]
        else:
            char_list += [f"un{name[2]}"]
            dir_list += [f"data/unipen_data_resized_refined/{file}"]

data = {"character": char_list, "image": dir_list}
df = pd.DataFrame(data=data)

df.to_csv("./unipen_data_resized_refined.csv", index=False)

