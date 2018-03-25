import os
import sys
from Bio import SeqIO 

''' Short script to take the provided fasta file and \
turn it into a nice manageable negatives.txt file (17 bp lines). '''

output = os.getcwd() + "/rap1-lieb-negatives.txt"
compare = sys.argv[2]
pieces = []
i = 0
records = list(SeqIO.parse(sys.argv[1], "fasta"))

# Read in all sequences
for r in records:
	i = 0
	aa = list(r.seq)
	if i+17 < len(aa):
		pieces.append(aa[i:i+17])
		i = i+17
print (len(pieces))

# Remove the ones that match a positive sequence
with open(compare) as f:
	lines = f.readlines()
print (len(lines), lines[0].split()[0], ''.join(pieces[0]))
for piece in pieces:
	if len(piece) == 17:
		piece = ''.join(piece)
		for line in lines:
			line = line.split()[0]
			if line == piece:
				print(line, piece)
				pieces.remove(piece)

# Write out the sequences to a text file in 17 nt increments.
file = open(output,"w") 
for piece in pieces:
	if len(piece) == 17:
		file.write(''.join(piece) + "\n")
file.close()

