import argparse


parser = argparse.ArgumentParser(description='merge multiple lst files')

parser.add_argument('--lst_in', dest='lst_in', help='directory containing all the lst files',
                    default=".", type=str)

pargs = parser.parse_args()

newlst = ""


l_idx=0

new_lst_str = ""

counter={}


with open(pargs.lst_in, 'r') as f:
    for line in f:

        words= line.split("\t")
        url=words[-1]
        words = words[3:len(words)-1]
        bboxes = [words[i:i+6] for i in range(0, len(words), 6)]

        for bbox in bboxes:
            counter[bbox[0]] = counter.setdefault(bbox[0],0) + 1

        l_idx+=1



print "number of images: {}".format(l_idx)

for k,v in counter.iteritems():
    print "class {}: {}".format(k, v)