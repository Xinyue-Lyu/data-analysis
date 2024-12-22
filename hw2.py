import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

col = ['r', 'b', 'g', 'c', 'm', 'k']

tol =  1e-4; K = 1; L = 4; xp = [-L,L]
xspan = np.arange(xp[0], xp[1]+0.1, 0.1)  #x value from -4 to 4

A1 = A1 = np.zeros((len(xspan), 5)) #eigenfunctions: ϕ_n
A2 = [] #eigenvalue: epsilon

def shoot(y, x, K, epsilon):
    return [y[1], (K*x**2 - epsilon) * y[0]]

epsilon_start = 0.1   
A = np.sqrt(K*L**2 - epsilon_start)       # y'(-4)=A
y0 = [1, A]                                    #initial guess y(-4)=1, y'(-4)=A

for modes in range(1, 6):
    epsilon = epsilon_start
    depsilon = 0.2

    for i in range(1000): # begin convergence loop for epsilon
        sol = odeint(shoot, y0, xspan,  args=(K,epsilon,))              #y1, y2

        if abs(sol[-1, 1] + np.sqrt(K*L**2 - epsilon)*sol[-1,0] ) < tol:   # sol[-1,0]-0 
            break

        if ((-1) ** (modes + 1) * (sol[-1, 1] + np.sqrt(K*L**2 - epsilon)*sol[-1,0])) > 0:  
            epsilon += depsilon
        else:
            epsilon -= depsilon/2
            depsilon /= 2
   
    A2.append(epsilon)
    epsilon_start = epsilon + 0.1


    norm = np.trapz(sol[:, 0] * sol[:, 0], xspan)  
    A1[:, modes-1] = abs(sol[:,0]/np.sqrt(norm))
    plt.plot(xspan, A1[:, modes-1], col[modes - 1])

plt.show()
#print(np.shape(A1))
print(A2)

