from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
parser.add_argument('--input', dest='input', help='train or val or test',
                    default="train", type=str)


pargs = parser.parse_args()

gidx=0
data=''
with open(pargs.input+'.lst', 'r') as f:
    lines=f.readlines()
    for line in lines:
        words=line.split('\t')
        words2=line.split('\t',1)
        path=words[len(words)-1].split()[0]
        try:
                Image.open(path).verify()
                newline=str(gidx)+'\t'+words2[1]
                gidx+=1
                data+=newline
        except:
            print 'faulty file: ' + path


with open(pargs.input+'_filtered.lst','w') as f:
    f.write(data)


