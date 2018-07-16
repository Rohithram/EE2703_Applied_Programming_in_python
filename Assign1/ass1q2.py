#Question 2
#Python code to create and print the list using given algorithm
# Name : Rohithram R
# ROLL NO : EE16B031

# WHAT THE PROGRAM DOES?

# The program aims to create a list by taking fractional part of previous value in sum with pi scaled to 100
# prints the ouput of the list which is formed using this algorithm in text file

# OUTPUT : Text File : 'q2outputforPython.txt' - contains the output of the program i.e list created with only 4 decimals

#Importing pi value from math library
from math import pi

#Creates empty list
n=[]
n.append(0.2000)                                #First value is appended to list
alpha = pi                                      #assign pi value to alpha
print("Output of question2 python\n")

for k in range(1,1000):
    temp = (n[k-1]+alpha)*100                   #storing the value in temporary variable 'temp'
    n.append((temp-int(temp)))                  #for getting fractional part: number - integer part of (number)
#     n[k] = round(n[k],4)
    
for k in range(len(n)):
    print "%d : %.4f" %(k,n[k])                 #printing the list with precision of 4 decimal places
