# load libraries and set plot parameters
from pylab import *
import scipy.signal as sp

plt.rcParams['savefig.dpi'] = 75

plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 12, 9
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.numpoints'] = 2
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = r"\usepackage{subdepth}, \usepackage{type1cm}"

'''
function to solve for x(t)
Arguments : x0    - x(0)
            x0dot - derivative of x at x=0
            decay - decay factor
            freq  - frequency at which spring operates(resonant case)
Returns   : t and x(t)
'''

def laplaceSolver(x0,x0dot,decay,freq):    
    Xnum = poly1d([1,decay])+polymul([x0,x0dot],[1,2*decay,(pow(freq,2)+pow(decay,2))])
    Xden = polymul([1,0,pow(freq,2)],[1,2*decay,(pow(freq,2)+pow(decay,2))])
    
    #Computes the impulse response of the transfer function
    Xs = sp.lti(Xnum,Xden)
    t,x=sp.impulse(Xs,None,linspace(0,100,10000))
    return t,x

# solving for two cases with decay of 0.5 and 0.05

t1,x1 = laplaceSolver(0,0,0.5,1.5)
t2,x2 = laplaceSolver(0,0,0.05,1.5)
t3,x3 = laplaceSolver(0,0,0.005,1.5)
t4,x4 = laplaceSolver(0,0,5,1.5)

#plot of x(t) with decay of 0.5

fig1a = figure()
ax1a = fig1a.add_subplot(111)
ax1a.plot(t1,x1,'b',label="decay = 0.5")
ax1a.legend()
title(r"Figure 1a: $x(t)$ of spring system")
xlabel(r"$t \to $")
ylabel(r"$x(t) \to $")
grid()
savefig("Figure1a.jpg")
show()


#plot of x(t) with decay values of 0.5 and 0.05
fig1 = figure()
ax1 = fig1.add_subplot(111)
ax1.plot(t1,x1,'b',label="decay = 0.5")
ax1.plot(t2,x2,'r',label="decay = 0.05")
ax1.plot(t3,x3,'g',label="decay = 0.005")
ax1.plot(t4,x4,'k',label="decay = 5")

ax1.legend()
title(r"Figure 1b: $x(t)$ of spring system as function of decay value of $f(t)$")
xlabel(r"$t \to $")
ylabel(r"$x(t) \to $")
grid()
savefig("Figure1b.jpg")
show()

'''
function to return f(t) for various parameters
Arguments : t     - time
            freq  - frequency at system is excited
            decay - decay factor
Returns   : t and x(t)
'''

def f(t,freq,decay):
    return cos(freq*t)*exp(-decay*t)


'''
function to solve for Transfer function H(s)
Arguments : x0    - x(0)
            x0dot - derivative of x at x=0
            decay - decay factor
            freq  - frequency at which system is excited
Returns   : Hs    - transfer function of the system
'''

def getTransferfunc(x0,x0dot,decay,freq):    
    
    #natural frequency is 1.5rad/s
    nat_freq = 1.5
    Hnum = poly1d([1,decay])+polymul([x0,x0dot],[1,2*decay,(pow(freq,2)+pow(decay,2))])+poly1d([1])
    Hden = polymul([1,0,pow(nat_freq,2)],[1,decay])
    
    #Computes the impulse response of the transfer function
    Hs = sp.lti(Hnum,Hden)
    return Hs


#Plot of x(t) with different input frequencies
fig2 = figure()
ax2 = fig2.add_subplot(111)
title(r"Figure 2: $x(t)$ of spring system with varying Frequency")

#For loop  to plot x(t) for different values of freq
for w in arange(1.4,1.6,0.05):
    decay = 0.05
    H = getTransferfunc(0,0,decay,w)
    t = linspace(0,200,10000)
    t,y,svec=sp.lsim(H,f(t,w,decay),t)
    legnd = "$w$ = %g rad/s"%(w) 
    ax2.plot(t,y,label=legnd) 
    ax2.legend()
    
xlabel(r"$t \to $")
ylabel(r"$x(t) \to $")
grid()
savefig("Figure2.jpg")
show()

#Calculating bode plot for transfer function H
w1,mag,phi=H.bode()

#Plot of x(t) with different input frequencies
fig3 = figure()
ax3 = fig3.add_subplot(111)
title("Figure 3: $|H(j\omega)|$ - Bode plot of the transfer function")

ax3.semilogx(w1,mag) 
xlabel(r"$\omega$")
ylabel(r"$ 20\log|H(jw)| \to $")
grid()
savefig("Figure3.jpg")
show()


'''
function to solve for Transfer function H(s)
Arguments : num_coeff   - array of coefficients of denominator polynomial
            den_coeff   - array of coefficients of denominator polynomial
Returns   : t,h         - time and response of the system
'''   

def coupledSysSolver(num_coeff,den_coeff):
    H_num = poly1d(num_coeff)
    H_den = poly1d(den_coeff)
    
    Hs = sp.lti(H_num,H_den)
    t,h=sp.impulse(Hs,None,linspace(0,20,1000))
    return t,h


#find x and y using above function
t1,x  = coupledSysSolver([1,0,2],[1,0,3,0])
t2,y = coupledSysSolver([2],[1,0,3,0])

#plot x(t) and y(t)
fig4 = figure()
ax4 = fig4.add_subplot(111)
ax4.plot(t1,x,'b',label="$x(t)$")
ax4.plot(t2,y,'r',label="$y(t)$")
ax4.legend()
title(r"Figure 4: Time evolution of $x(t)$ and $y(t)$ for $0 \leq t \leq 20$. of Coupled spring system ")
xlabel(r"$t \to $")
ylabel(r"$x(t),y(t) \to $")
grid()
savefig("Figure4.jpg")
show()

'''
function to solve for Transfer function H(s)
Arguments : H         - Transfer function.
Returns   : w,mag,phi
'''   

def CalcMagPhase(H):
    w,mag,phi=H.bode()
    return w,mag,phi


'''
function to solve given RLC network for any R,L,C values
Returns   : w,mag,phi,Hs
'''  

def RLCnetwork(R,C,L):
    Hnum = poly1d([1])
    Hden = poly1d([L*C,R*C,1])
    
    #Computes the impulse response of the transfer function
    Hs = sp.lti(Hnum,Hden)
    #Calculates magnitude and phase response
    w,mag,phi = CalcMagPhase(Hs)
    return w,mag,phi,Hs


#Finds magnitude and phase response of Transfer function
R = 100
L = 1e-6
C = 1e-6
w,mag,phi,Hrlc = RLCnetwork(R,L,C)

#plot Magnitude Response 
fig5 = figure()
ax5 = fig5.add_subplot(111)
ax5.semilogx(w,mag,'b',label="$Mag Response$")
ax5.legend()
title(r"Figure 5: Magnitude Response of $H(jw)$ of Series RLC network")
xlabel(r"$ \log w \to $")
ylabel(r"$ 20\log|H(jw)|  \to $")
grid()
savefig("Figure5.jpg")
show()

#Plot of phase response
fig6 = figure()
ax6 = fig6.add_subplot(111)
ax6.semilogx(w,phi,'r',label="$Phase Response$")
ax6.legend()
title(r"Figure 6: phase response of the $H(jw)$ of Series RLC networkfor")
xlabel(r"$ \log w \to $")
ylabel(r"$ \angle H(j\omega)$ $\to $")
grid()
savefig("Figure6.jpg")
show()

'''
function to return vi(t)
arguments : t   - time variable
            w1  - frequency of 1st cos term
            w2  - frequency of 2nd cos term
Returns   : vi(t)
'''  

def vi(t,w1,w2):
    return cos(w1*t)-cos(w2*t)


#Defines time from 0 to 90 msec
t  = linspace(0,90*pow(10,-3),pow(10,6))
#finding vi(t) using above function
Vi = vi(t,pow(10,3),pow(10,6))

#finds Vo(t) using lsim
t,Vo,svec=sp.lsim(Hrlc,Vi,t)
vo_ideal = cos(1e3*t)


#plot of Vo(t) for large time i.e at steady state
#Long term response
fig7a = figure()
ax7a = fig7a.add_subplot(111)
ax7a.plot(t,Vo,'r',label="Output Voltage $v_0(t)$ for large time")
ax7a.legend()
title(r"Figure 7a: Output Voltage $v_0(t)$  of series RLC network for given $v_i(t)$ at Steady State")
xlabel(r"$ t \to $")
ylabel(r"$ y(t) \to $")
grid()
savefig("Figure7a.jpg")
show()

#plot of Vo(t) for large time i.e at steady state
#Long term response
fig7 = figure()
ax7 = fig7.add_subplot(111)
ax7.plot(t,Vo,'r',label="Output Voltage $v_0(t)$ - zoomed in ")
ax7.plot(t,vo_ideal,'g',label="Ideal Low Pass filter Output with cutoff at $10^4$")
xlim(0.0505,0.051)
ylim(0.75,1.1)
ax7.legend()
title(r"Figure 7b: Output Voltage $v_0(t)$  Vs Ideal Low pass filter Output")
xlabel(r"$ t \to $")
ylabel(r"$ y(t) \to $")
grid()
savefig("Figure7b.jpg")
show()

#Plot of Vo(t) for 0<t<30usec
fig8 = figure()
ax8 = fig8.add_subplot(111)
ax8.plot(t,Vo,'r',label="Output Voltage $v_0(t)$ : $0<t<30\mu sec$")
ax8.legend()
title(r"Figure 8: Output Voltage $v_0(t)$ for $0<t<30\mu sec$")
xlim(0,3e-5)
ylim(-1e-5,0.3)
xlabel(r"$ t \to $")
ylabel(r"$ v_0(t) \to $")
grid()
savefig("Figure8.jpg")
show()
