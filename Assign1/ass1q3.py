#Question 3
#Python code to count frequency of words in a text file
# Name : Rohithram R
# ROLL NO : EE16B031

# WHAT THE PROGRAM DOES?

# The program aims to count the frequency of words in a given text file using dictionaries
# prints the ouput for words vs frequency and plot a sample histogram for first 15 words in
# descending order of their frequency, to gain an insight how frequency is distributed over words!


# INPUT : Text File : 'q3file.txt'    - contains the Hound of Baskervilles story
# OUTPUT : Text File : 'q3output.txt' - contains the output of the program i.e words vs frequency
#          Histogram : 'q3histogram.png' - contains the histogram for first 15 high frequency words for sample
# Only 15 words are plotted due to lack of computation power and for visibility

#Importing Regex and pylab libraries

import re
from pylab import *

#Reading the input text file
with open('q3file.txt','r') as f:
    contents=f.read()               #Read the whole file as a string and stored it in contents

# regex pattern for filtering punctuations from words
#hence allowing only alphanumeric characters

pattern = re.compile(r'[^a-zA-Z0-9 ]')
words = contents.split()

#function to find the frequency of words in given text file

def freqOfWords(words):
    for w in words:
        w = pattern.sub('',w)       #Removing the unwanted punctuations from words
        if(w in d):
            d[w]+=1                 #Increment the count if word is already present
        else:
            d[w]=1                  #Set frequency to 1 if a word is newly found

    return d                        # returns the dictionary


# creates an empty dictionary
d={} 
d = freqOfWords(words)              #Function call to find frequency of words
freq = []                           #Creates empty list for storing frequency and words separately
words_sorted = []                   #To plot histogram

print("Output of question3\n")


print("Words | Frequency")

#frequencies sorted in descending order

for k in sorted(d,key=d.get,reverse=True):  
    words_sorted.append(k)          #sorted words appended to list to be labels in x-axis of histogram
    freq.append(d[k])               #Sorted frequencies are appended to list to be y-axis of histogram
    print " %s : %d" %(k,d[k])      #printing the words vs frequency as output

 
Wordnos = 1 + np.arange(len(words_sorted))    #used to give numberings for words, utilised for histogram

#Plotting histogram of first 15 words in descending order of frequencies!

bar(Wordnos[0:15],freq[0:15], align='center')
xticks(Wordnos[0:15],words_sorted[0:15])
title("Word Frequency Histogram of First 15 words")
xlabel("words")
ylabel("Frequency")
show()
                                            #End of the program