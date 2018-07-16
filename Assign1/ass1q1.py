#Question 1
#Python code to print first 10 fibonacci numbers
# Name : Rohithram R
# ROLL NO : EE16B031

# WHAT THE PROGRAM DOES?

# The program aims to print first 10 fibonacci numbers
# prints the ouput in terminal 

# OUTPUT : Screen Shot of Terminal : 'q1outpython.png' - contains the output of the program

 #Assigning first two numbers of fibonacci series as 1
n=1
nold=1                                 
print("Output of question1 in python\n")

print 1,nold                                #printing first two numbers in the series
print 2,n

for k in range(3,11):
    new =  n + nold                         #Assigning next value in fibonacci series as previous + current value
    nold = n                                #updating previous value to current
    n = new                                 #updating current value to next value
    print k,new                             #printing the fibonacci numbers
                            