// Question 1
// C code to print first 10 fibonacci numbers
// Name : Rohithram R
// ROLL NO : EE16B031

// WHAT THE PROGRAM DOES?

// The program aims to print first 10 fibonacci numbers
// prints the ouput in terminal 

// OUTPUT : Screen Shot of Terminal : 'q1outpython.png' - contains the output of the program


//Listing of required header files.
#include<stdio.h>                     
#include<stdlib.h>

int main()              
    {       
    int n=1,nold=1,new=0,k;                             // Assigning first two numbers of fibonacci series as 1
    printf("Output of question1 in c language\n");
    printf("1 : %d \n",nold);
    printf("2 : %d \n",n);
    for(k=3;k<11;k++){
        new = n+nold;                                    // Assigning next value in fibonacci series as previous + current value
        nold = n;                                        //updating previous value to current
        n = new;                                         //updating current value to next value      
        printf("%d : %d\n",k,new);
    }
    return 0;
}
//End of the program.....
