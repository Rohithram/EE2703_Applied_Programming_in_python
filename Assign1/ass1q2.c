//Question 2
// C code to create and print the list using given algorithm
// Name : Rohithram R
// ROLL NO : EE16B031
// WHAT THE PROGRAM DOES?
// The program aims to create a list by taking fractional part of previous value in sum with pi scaled to 100
// prints the ouput of the list which is formed using this algorithm in text file
// OUTPUT : Text File : 'q2outputforCcode.txt' - contains the output of the program i.e list created with only 4 decimals

//Listing of required header files.
#include<stdio.h>                     
#include<math.h>

int main()              
    {       
    double n[1000];
    double alpha = 3.14159265358979323846;  //pi value 
    double temp=0.0,integer;
    int k;
    n[0]=0.2000;                                    //storing 1st value as 0.2000
    printf("Output of question2 in c language\n");

    for(k=1;k<1000;k++){
        temp = (n[k-1]+alpha)*100;                 //storing the value in temporary variable 'temp'
        n[k]=modf(temp,&integer);                  //modf is math library function which takes fractional part
                                                   //of a floating point number, it takes number and integer part
                                                   //as arguments
    }
    //for loop to print the list                           
    for (k=0;k<1000;k++){
        printf ("%d : %.4f \n",k,n[k]);    
    }
    return 0;
}
//End of the program.....
