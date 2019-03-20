import json
with open("imagenet_class_index.json", "r") as f:
    data = f.read()
x = data
y = json.loads(x)
print (y["999"])
i = 1000
with open("synset.txt", "r") as f:
    for line in f:
        sysdata = line.strip().split(" ")
        print (i, " ", sysdata)
        i=i+1

