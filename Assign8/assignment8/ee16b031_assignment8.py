# load libraries and set plot parameters

import scipy.signal as sp
from pylab import *
from sympy import *

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
plt.rcParams['savefig.dpi'] = 75

plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 12, 9
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.numpoints'] = 2
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = r"\usepackage{subdepth}, \usepackage{type1cm}"

'''
function to solve for V(s) by Matrix inversion
This function used for Low pass filter
arguments : R1,R2,C1,C2,G   - parameters of the circuit
            Vi - Laplace transform of Input.
'''   

def LpfResponse(R1,R2,C1,C2,G,Vi):
    s=symbols('s')
    A=Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0],[0,-G,G,1],
              [-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b=Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    return (A,b,V)


'''
function to solve for Transfer function H(s)
To convert sympy polynomial to sp.lti polynomial
Arguments : num_coeff   - array of coefficients of denominator polynomial
            den_coeff   - array of coefficients of denominator polynomial
Returns   : Hs          - Transfer function in s domain
'''   

def SympyToLti(num_coeff,den_coeff):
    num_coeff = np.array(num_coeff, dtype=float)
    den_coeff = np.array(den_coeff,dtype=float)
    H_num = poly1d(num_coeff)
    H_den = poly1d(den_coeff)
    Hs = sp.lti(H_num,H_den)
    return Hs


'''
function to solve for Output voltage for given circuit
Arguments : R1,R2,C1,C2,G   - parameters of the circuit
            Vi - Laplace transform of input Voltage
            circuitResponse - function defined which is either lpf or Hpf
Returns   : v,Vlti         - v is array of values in jw domain
                           - Vlti is sp.lti polynomial in s
'''   

def solver(R1,R2,C1,C2,G,Vi,circuitResponse):
    s = symbols('s')
    A,b,V = circuitResponse(R1,R2,C1,C2,G,Vi)
    Vo = V[3]
    num,den = fraction(simplify(Vo))
    num_coeffs = Poly(num,s).all_coeffs()
    den_coeffs = Poly(den,s).all_coeffs()
    Vlti = SympyToLti(num_coeffs,den_coeffs)
    
    w = logspace(0,8,801)
    ss = 1j*w
    hf = lambdify(s,Vo,"numpy")
    v = hf(ss)
    
    #Calculating Quality factor for the system
    if(Vi == 1):          # Vi(s)=1 means input is impulse
        Q  = sqrt(1/(pow(den_coeffs[1]/den_coeffs[2],2)/(den_coeffs[0]/den_coeffs[2])))
        print("Quality factor of the system : %g"%(Q))
        return v,Vlti,Q
    else:   
        return v,Vlti


#Declaring params of the circuit1
R1 = 10000
R2 = 10000
C1 = 1e-9
C2 = 1e-9
G  = 1.586

# w is x axis of bode plot
s = symbols('s')
w = logspace(0,8,801)

Vi_1 = 1     #Laplace transform of impulse
Vi_2 = 1/s   #Laplace transform of u(t)

#Finding Vo(t) for these given two inputs
Vo1,Vs1,Q = solver(R1,R2,C1,C2,G,Vi_1,LpfResponse)

# To find Output Voltage in time domain
t1,Vot1 = sp.impulse(Vs1,None,linspace(0,1e-2,10000))

Vo2,Vs2 = solver(R1,R2,C1,C2,G,Vi_2,LpfResponse)

# To find Output Voltage in time domain
t2,Vot2 = sp.impulse(Vs2,None,linspace(0,1e-3,100000))
#plot of Magnitude response of Transfer function

fig1 = figure()
ax1 = fig1.add_subplot(111)
ax1.loglog(w,abs(Vo1))
title(r"Figure 1a: $|H(j\omega)|$ : Magnitude response of Transfer function")
xlabel(r"$\omega \to $")
ylabel(r"$ |H(j\omega)| \to $")
grid()
savefig("Figure1a.jpg")
show()

#plot of unit Step response

fig1b = figure()
ax1b = fig1b.add_subplot(111)
# Input - Unit step function 
ax1b.step([t2[0],t2[-1]],[0,1],label=r"$V_{i}(t) = u(t)$")
ax1b.plot(t2,Vot2,label=r"Response for $V_{i}(t) = u(t)$")
ax1b.legend()
title(r"Figure 1b: $V_{o}(t)$ : Unit Step response in time domain")
xlabel(r"$t \to $")
ylabel(r"$ V_{o}(t) \to $")
grid()
savefig("Figure1b.jpg")
show()


#input sinusoid frequencies in rad/s
w1 = 2000*pi
w2 = 2*1e6*pi

#Laplace transform of given input sinusoid
Vi_3 =  w1/(s**2 + w1**2) + s/(s**2 + w2**2)
Vo3,Vs3 = solver(R1,R2,C1,C2,G,Vi_3,LpfResponse)
#Vo(t) for sinusoid input
t3,Vot3 = sp.impulse(Vs3,None,linspace(0,1e-2,10000))
#plot of Magnitude Response of sinusoidal input
    
fig2 = figure()
ax2 = fig2.add_subplot(111)
ax2.loglog(w,abs(Vo3))
title(r"Figure 2a: $|Y(j\omega)|$ : Magnitude Response for input sinusoid")
xlabel(r"$\omega \to $")
ylabel(r"$ |Y(j\omega)| \to $")
grid()
savefig("Figure2a.jpg")
show()

#plot of Vo(t) for sinusoidal input
fig2b = figure()
ax2b = fig2b.add_subplot(111)
ax2b.plot(t3,(Vot3))
ax2b.legend()
title(r"Figure 2b: $V_{o}(t)$ : Output Voltage for sinusoidal input")
xlabel(r"$t \to $")
ylabel(r"$ V_{o}(t) \to $")
grid()
savefig("Figure2b.jpg")
show()


'''
function to solve for V(s) by Matrix inversion
This function used for High pass filter
arguments : R1,R3,C1,C2,G   - parameters of the circuit
            Vi - Laplace transform of Input.
'''   

def HpfResponse(R1,R3,C1,C2,G,Vi):
    s = symbols('s')
    A=Matrix([[0,0,1,-1/G],
              [-s*C2*R3/(1+s*R3*C2),1,0,0],
              [0,-G,G,1],
              [(-1-(s*R1*C1)-(s*R3*C2)),s*C2*R1,0,1]])
    b=Matrix([0,0,0,-Vi*s*C1*R1])
    V = A.inv()*b
    return (A,b,V)


#Params for 2nd circuit
R1b = 10000
R3b = 10000
C1b= 1e-9
C2b = 1e-9
Gb = 1.586

#input frequencies for damped sinusoids
w1 = 2000*pi
w2 = 2e6*pi
#Decay factor for damped sinusoid
a = 1e5

Vi_1b = 1   # Laplace transform of impulse

#Laplace transform of damped sinusoid
Vi_2b =  w1/((s+a)**2 + w1**2) + (s+a)/((s+a)**2 + w2**2)
#Laplace of unit step
Vi_3b = 1/s

#Laplace transform of undamped input sinusoid
Vi_4b =  w1/(s**2 + w1**2) + s/(s**2 + w2**2)


'''
Solving for Output voltage for these inputs
Qb is the quality factor of this system
'''

Vo1b,Vs1b,Qb = solver(R1b,R3b,C1b,C2b,Gb,Vi_1b,HpfResponse)
t1b,Vot1b = sp.impulse(Vs1b,None,linspace(0,1e-2,10000))

Vo2b,Vs2b = solver(R1b,R3b,C1b,C2b,Gb,Vi_2b,HpfResponse)
t2b,Vot2b = sp.impulse(Vs2b,None,linspace(0,5e-5,1000001))

Vo3b,Vs3b = solver(R1b,R3b,C1b,C2b,Gb,Vi_3b,HpfResponse)
t3b,Vot3b = sp.impulse(Vs3b,None,linspace(0,5e-4,10001))

Vo4b,Vs4b = solver(R1b,R3b,C1b,C2b,Gb,Vi_4b,HpfResponse)
t4b,Vot4b = sp.impulse(Vs4b,None,linspace(0,1e-1,10000))

#plot of Magnitude response of Transfer function
fig3 = figure()
ax3 = fig3.add_subplot(111)
ax3.loglog(w,abs(Vo1b))
ax3.legend()
title(r"Figure 3: $|H(j\omega)|$ : Magnitude response of Transfer function")
xlabel(r"$\omega \to $")
ylabel(r"$ |H(j\omega)| \to $")
grid()
savefig("Figure3.jpg")
show()


#plot of Vo(t) for damped sinusoidal input
fig6a = figure()
ax6a = fig6a.add_subplot(111)
ax6a.plot(t4b,(Vot4b),label=r"Response for $V_{i}(t) = $ undamped sinusoid")
ax6a.legend()
title(r"Figure 6a: $V_{o}(t)$ : Output Voltage for undamped sinusoid input through High Pass filter")
xlabel(r"$t \to $")
ylabel(r"$ V_{o}(t) \to $")
grid()
savefig("Figure6a.jpg")
show()

#plot of Vo(t) for damped sinusoidal input

fig6 = figure()
ax6 = fig6.add_subplot(111)
ax6.plot(t2b,(Vot2b),label=r"Response for $V_{i}(t) = $ damped sinusoid")
ax6.legend()
title(r"Figure 6b: $V_{o}(t)$ : Output Voltage for damped sinusoid input")
xlabel(r"$t \to $")
ylabel(r"$ V_{o}(t) \to $")
grid()
savefig("Figure6b.jpg")
show()

#Plot vo(t)  for unit step input
fig7 = figure()
ax7 = fig7.add_subplot(111)
# Input - Unit step function 
ax7.step([t3b[0],t3b[-1]],[0,1],label = r"$V_{i}(t) = u(t)$")
ax7.plot(t3b,(Vot3b),label=r"Unit Step Response for $V_{i}(t) = u(t)$")
ax7.legend()
title(r"Figure 7: $V_{o}(t) $ : Unit step response in time domain")
xlabel(r"$ t (seconds) \to $")
ylabel(r"$ V_{o}(t) \to $")
grid()
savefig("Figure7.jpg")
show()
