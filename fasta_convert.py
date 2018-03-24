import os
import sys
from Bio import SeqIO 

''' Short script to take the provided fasta file and \
turn it into a nice manageable negatives.txt file (17 bp lines). '''

output = os.getcwd() + "/rap1-lieb-negatives.txt"
pieces = []
i = 0
records = list(SeqIO.parse(sys.argv[1], "fasta"))

for r in records:
	i = 0
	aa = list(r.seq)
	if i+17 < len(aa):
		pieces.append(aa[i:i+17])
		i = i+17
print (len(pieces), len(pieces[-1]))

# Write out the sequences to a text file in 17 nt increments.
file = open(output,"w") 
for piece in pieces:
	if len(piece) == 17:
		file.write(''.join(piece) + "\n")
file.close()

