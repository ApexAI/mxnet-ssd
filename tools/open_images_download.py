import csv


classes={}

with open('/mnt/data/dataset_stuff/openimages/class-descriptions.csv', 'r') as f:
    reader=csv.reader(f,delimiter=',')
    for row in reader:
        classes[row[0]]=row[1]



outdata=""
lines=[]
with open('/mnt/data/dataset_stuff/openimages/classes-bbox-trainable.txt', 'r') as f:
    for line in f.readlines():
        dat=line.strip()
        outdata += dat + "," + classes[dat] + "\n"




with open('/mnt/data/dataset_stuff/openimages/classes-bbox-trainable_labels.csv', 'w') as f:
    f.write(outdata)