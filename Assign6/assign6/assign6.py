
# load libraries and set plot parameters
from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
import sys
from  tabulate import tabulate


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
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.numpoints'] = 2
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = r"\usepackage{subdepth}, \usepackage{type1cm}"


# To get the arguments using sys.argv from command line
if(len(sys.argv)==7):
    n,M,Msig,nk,u0,p = sys.argv[1:]
else:
    n=100                   # spatial grid size.
    M=5                     # number of electrons injected per turn.
    Msig = 2                #Standard deviation of injected electrons
    nk=500                  # number of turns to simulate.
    u0=5                    # threshold velocity.
    p=0.25                  # probability that ionization will occur

n=int(n)                   
M=int(M)                   
Msig =int(Msig)               
nk=int(nk)           
u0=int(u0)               
p=float(p)            

#Initialsing all these arrays with zeros with size nM
N = n*M
xx = np.zeros(N)
u = np.zeros(N)
dx = np.zeros(N)

# Declare intensity and electron position and velocity 
I = []
X = []
V = []


# Function to find the electrons inside the tubelight
def findexistElectrons(xx):
    #ii is a vector containing the indices of vector xx 
    #that have positive entries.
    ii = where(xx>0)
    return ii[0]    



# function to upate the Velocity,Displacement,position of electrons to zero
# of electrons which hit anode i.e its position > n(Outside tubelight)
def updatePosVel(xx,u,dx):
    indexes = where(xx>n)
    xx[indexes] = 0
    u[indexes] = 0
    dx[indexes] = 0
    return xx,u,dx


# function to find the energetic electrons inside tubelight
# by checking its velocity is more than threshold and returns the indices
# of those electrons
def findEnergeticElectrons(u):
    kk = where(u >= u0)[0]
    ll = where(rand(len(kk))<=p)
    kl = kk[ll]
    return kl


# function to inject electrons by finding out indexes of electrons 
# whose position is less than 0 .
def toInjectElectron(xx,m):
    inj = where(xx <= 0)
    return inj


# For loop to run the simulation nk times
for j in range(1,nk):

    ii = findexistElectrons(xx)      #to find electrons inside tubelight
    X.extend(xx[ii].tolist())        #Storing active electrons in X each turn
    V.extend(u[ii].tolist())         #Storing velocities of these electrons

    dx[ii] = u[ii] + 0.5             #displacement of electron from kinematics
    xx[ii] = xx[ii] + dx[ii]         #updating position by adding the displacement
    u[ii]  = u[ii] + 1               #updating the velocity at new position
    xx,u,dx = updatePosVel(xx,u,dx)  #update position,velocity of electrons which hit anode
    kl = findEnergeticElectrons(u)   #indexes of energetic electrons
    u[kl] = 0                        #set velocity of energetic electrons to zero
    xx[kl] = xx[kl]-dx[kl]*random()  #updating position after collision

    I.extend(xx[kl].tolist())        #Storing ionized electrons positions
    m=int(randn()*Msig+M)            #Actual no of injected electrons
    inj = toInjectElectron(xx,m)     #indexes where to inject electrons
    xx[inj] = 1                      #Set their position with 1


fig1 = figure()
ax1 = fig1.add_subplot(111)
ax1.hist(X,n, alpha=0.9, histtype='bar', ec='black')
ax1.legend()
xticks(range(0,n+1,10))
title(r"Figure 1: Histogram of electron density")
xlabel("$n$")
ylabel("No of Electrons")
grid()
savefig("Figure1.jpg")


fig2 = figure()
ax2 = fig2.add_subplot(111)
Idata = ax2.hist(I,n,alpha=0.9, histtype='bar', ec='black')
ax2.legend()
xticks(range(0,n+1,10))
title(r"Figure 2: Emission Intensity along the length of Tubelight")
xlabel("$n$")
ylabel("I")
grid()
savefig("Figure2.jpg")


fig3 = figure()
ax3 = fig3.add_subplot(111)
ax3.plot(X,V,'go')
ax3.legend()
title(r"Figure 3: Electron Phase space plot")
xlabel("$x$")
xticks(range(0,n+1,10))
ylabel("$v$")
grid()
show()
savefig("Figure3.jpg")


bins = Idata[1]
xpos =0.5*(bins[0:-1]+bins[1:])

table = zip(xpos,Idata[0])

headers = ["Position","Count"]
#tabulating Intensity of electrons Vs position inside the tubelight
print("Intensity data:")
print(tabulate(table,tablefmt="fancy_grid",headers=headers))
