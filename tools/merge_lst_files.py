
import argparse
import os
import random


parser = argparse.ArgumentParser(description='merge multiple lst files')

parser.add_argument('--lstdir', dest='lstdir', help='directory containing all the lst files',
                    default=".", type=str)

parser.add_argument('--out', dest='out', help='output lst file',
                    default=".", type=str)

pargs = parser.parse_args()


filenames=[ os.path.join(pargs.lstdir,file) for file in os.listdir(pargs.lstdir)]


lst_data=""


lines=[]

for filename in filenames:
    with open(filename,'r') as f:
        lines+=f.readlines()


random.shuffle(lines)

for lst_idx, line in enumerate(lines):
    lst_data += "{}\t{}".format(lst_idx, line.split('\t',1)[1])


with open(pargs.out,'w') as f:
    f.write(lst_data)




