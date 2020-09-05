

import os 

dir=r"D:\item\InceptionV3_model\st_table.txt"
with open(dir,"r",encoding="utf-8") as f:
    list=f.readlines()
dict={}
for i in list:
    a=i.strip().split(" ")
    dict[a[0]]=a[1]

new_dict = {v : k for k, v in dict.items()}
print(new_dict["大"])