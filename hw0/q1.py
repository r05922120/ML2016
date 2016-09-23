import sys
import os
index = sys.argv[1]
index = int(index)
filename = sys.argv[2]
file = open(filename,'r')
numbers = []
output = ""

for line in file:
  temp = line.strip('\n').split(' ')
  temp.remove('')
  numbers.append(float(temp[index]))

file.close()
numbers = sorted(numbers)

for i in numbers:
  output=output+str(i)+","

output = output[0:len(output)-1]

file2 = open("ans1.txt",'w')
file2.write(output)
file2.close()
