import argparse
import os
import random
parser = argparse.ArgumentParser(description='Train a Single-shot detection network')

parser.add_argument('--seqdir', dest='seqdir', help='directory containing all the corrected sequences',
                    default="../../../datasets/apex_tl/images", type=str)

parser.add_argument('--labeldir', dest='labeldir', help='directory containing all the corrected sequences',
                    default="../../../datasets/apex_tl/labels", type=str)
parser.add_argument('--output', dest='output', help='directory containing all the corrected sequences',
                    default="apex_combined_0602", type=str)
parser.add_argument('--valsplit', dest='valsplit', help='split ratio',
                    default=0.2, type=float)
pargs = parser.parse_args()

seqs=os.listdir(pargs.seqdir)

train_lst=""
val_lst=""
val_idx=0
lines=[]
for seq in seqs:
    path='images/'+seq+"/"
    with open(os.path.join(pargs.labeldir,seq+".lst"), 'r') as f:
        for line in f:
            words=line.split('\t')
            #print words

            lines.append("\t"+words[1]+"\t"+path + os.path.basename(words[2]))


num_samples=len(lines)
random.shuffle(lines)
train_num=int(num_samples*(1-pargs.valsplit))
val_num=num_samples-train_num

for i in range(train_num):
    train_lst+=str(i)+lines[i]


with open(pargs.output+"_train.lst", "w") as text_file:
    text_file.write(train_lst)

for i in range(train_num,num_samples):
    val_lst += str(val_idx) + lines[i]
    val_idx+=1

with open(pargs.output + "_val.lst", "w") as text_file:
    text_file.write(val_lst)