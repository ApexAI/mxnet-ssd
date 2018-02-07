import argparse
import os

parser = argparse.ArgumentParser(description='Train a Single-shot detection network')

parser.add_argument('--seqdir', dest='seqdir', help='directory containing all the corrected sequences',
                    default="../../../datasets/apex_tl/images", type=str)

parser.add_argument('--labeldir', dest='labeldir', help='directory containing all the corrected sequences',
                    default="../../../datasets/apex_tl/labels", type=str)
parser.add_argument('--output', dest='output', help='directory containing all the corrected sequences',
                    default="apex_combined.lst", type=str)
pargs = parser.parse_args()

seqs=os.listdir(pargs.seqdir)

lst_data=""
global_idx=0
for seq in seqs:
    path='images/'+seq+"/"
    with open(os.path.join(pargs.labeldir,seq+".lst"), 'r') as f:
        for line in f:
            words=line.split('\t')
            #print words

            lst_data+=str(global_idx)+"\t"+words[1]+"\t"+path + os.path.basename(words[2])
            global_idx+=1

with open(pargs.output, "w") as text_file:
    text_file.write(lst_data)
