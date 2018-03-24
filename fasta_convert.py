import os
import sys
from Bio import SeqIO 

''' Short script to take the provided fasta file and turn it into a nice manageable negatives.txt file (17 bp lines). '''

output = os.getcwd() + "/rap1-lieb-negatives.txt"
pieces = []
i = 0
records = list(SeqIO.parse(sys.argv[1], "fasta"))

for r in records:
	aa = list(r.seq)
	j = i+17
	if j < len(aa):
		pieces.append(aa[i:j])
		i = j
print (len(pieces), len(pieces[0]))

file = open(output,"w") 
for piece in pieces:
	if len(piece) == 17:
		file.write(''.join(piece) + "\n")
file.close()

