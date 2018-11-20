import argparse
import os
import random
parser = argparse.ArgumentParser(description='split a lst file into train and val files')

parser.add_argument('--prefix', dest='prefix', help='lst prefix',
                    default="", type=str)
parser.add_argument('--ratio', dest='ratio', help='split ratio',
                    default=0.85, type=float)
pargs = parser.parse_args()


train_lst=""
val_lst=""
val_idx=0
lines=[]

with open(pargs.prefix+'.lst', 'r') as f:
    for line in f:
        words=line.split('\t',1)
        lines.append(words[1])


num_samples=len(lines)
random.shuffle(lines)
train_num=int(num_samples*(pargs.split))
val_num=num_samples-train_num

for i in range(train_num):
    train_lst+=str(i)+"\t"+lines[i]

with open(pargs.prefix+"_train.lst", "w") as text_file:
    text_file.write(train_lst)

for i in range(train_num,num_samples):
    val_lst += str(val_idx)+ "\t" + lines[i]
    val_idx+=1

with open(pargs.prefix+ "_val.lst", "w") as text_file:
    text_file.write(val_lst)