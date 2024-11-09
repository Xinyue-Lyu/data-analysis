import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

#Part A
tol = 1e-4
L = 4
xspan = np.arange(-L, L + 0.1, 0.1)
N = len(xspan)

Ea = 0.1  # Starting guess for epsilon
Esola = np.zeros(5)  # Store eigenvalues
ysola = np.zeros((N, 5))  # Store eigenfunctions

# Define the shooting function
def shoot(x, y, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

for modes in range(1,6):
    dE = 0.2
    for j in range(1000):
        y0 = [1, np.sqrt(L**2 - Ea)]  # Initial conditions
        sol = solve_ivp(lambda x, y: shoot(x, y, Ea), [xspan[0], xspan[-1]], y0, t_eval=xspan)

        ys = sol.y.T  # `ys` is now correctly assigned the transpose

        # Boundary condition check
        if abs(ys[-1, 1] + np.sqrt(L**2 - Ea) * ys[-1, 0]) < tol:
            break
        if ((-1) ** (modes + 1) * (ys[-1, 1] + np.sqrt(L**2 - Ea) * ys[-1, 0])) > 0:
            Ea += dE
        else:
            Ea -= dE
            dE /= 2  # Refine step size

    Esola[modes-1] = Ea  # Store the found eigenvalue

    # Normalize the eigenfunction
    norm = np.sqrt(np.trapezoid(ys[:, 0]**2, xspan))  # Calculate normalization factor
    ysola[:, modes-1] = np.abs(ys[:, 0] / norm)  # Store normalized eigenfunction
    
    Ea += 0.2  # Update starting epsilon for the next mode

A1 = ysola  # Eigenfunctions
A2 = Esola  # Eigenvalues

#Part b =============================================================================
dx = xspan[1]-xspan[0]
A = np.zeros((N-2,N-2))

for j in range(N-2):
    A[j][j] = -2-(xspan[j+1] * dx)**2
    if j < N-3:
         A[j+1][j] = 1
         A[j][j+1] = 1

A[0][0] = A[0][0] + 4/3
A[0][1] = 1 - 1/3
A[-1][-2] = 1 - 1/3
A[-1][-1] = A[-1][-1] + 4/3

eigvals,eigvers = eigs(-A, k = 5, which = 'SM')
v2 = np.vstack([4*eigvers[0,:]/3 - eigvers[1,:]/3, eigvers, 
                4*eigvers[-1,:]/3 - eigvers[-2,:]/3])

ysolb = np.zeros((N,5))
Esolb = np.zeros(5)

for j in range(5):
     norm = np.trapezoid(v2[:, j] ** 2, xspan)#calculate norm
     ysolb[:,j] = abs(v2[:,j]/np.sqrt(norm))
Esolb = eigvals[:5]/(dx**2)

A3 = ysolb
A4 = Esolb
#print(A3)
#print(A4)

#PartC===============================================================
L = 2
x = np.arange(-L, L+0.1, 0.1)
n = len(x)

A5 = np.zeros((n,2))
A6 = np.zeros(2)
A7 = np.zeros((n,2))
A8 = np.zeros(2)

def rhs_c(x,y,Ec, gamma):
     return [y[1], (gamma * y[0]**2 + x**2 - Ec) * y[0]]

for gamma in [0.05, -0.05]: 
    E0 = 0.1
    Am = 1e-6  # Initial guess for E and amplitude

    for jmodes in range(2):
        da = 0.01
        
        for jj in range(100):   # Adjust E until convergence
            Ec = E0
            dE = 0.2

            for j in range(100):
                y0 = [Am, np.sqrt(L**2 - Ec) * Am]
                sol_c = solve_ivp(lambda x, y: rhs_c(x,y,Ec,gamma), [x[0],x[-1]], y0, t_eval = x)
                ys_c = sol_c.y.T
                xs = sol_c.t
                
                bc = ys_c[-1, 1] + np.sqrt(L**2 - Ec) * ys_c[-1,0]

                if abs(bc) < tol:  
                    break

                if (-1) ** jmodes * bc > 0 :  #adjust epsilon
                    Ec += dE
                else: 
                    Ec -= dE/2
                    dE /=2

            # Normalize eigenfunction
            area = np.trapezoid(ys_c[:, 0]**2, xs)
            if abs(area-1) < tol:
                break

            #adjust amplitude
            if area < 1:
                Am += da
            else:
                Am -= da
                da /= 2
        
        # Save eigenvalues and eigenfunctions
        if gamma > 0:
            A6[jmodes] = Ec
            A5[:, jmodes] = np.abs(ys_c[:, 0])  # Save eigenfunctions, gamma=0.05
        else:
            A8[jmodes] = Ec
            A7[:, jmodes] = np.abs(ys_c[:, 0]/np.sqrt(area)) 
        
         # Update E0 for next mode
        E0 = Ec + 0.2

#PartD =============================================================
L = 2
xd = [-L, L]; E_d = 1
y0 = [1, np.sqrt(L**2-E_d)]; 
tols= [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]

def rhs_a(x, y, E_d):
    return [y[1], (x**2-E_d)*y[0]]

step_sizes_rk45 = []
step_sizes_rk23 = []
step_sizes_radau = []
step_sizes_bdf = []

for tol in tols:
    options = {'rtol': tol, 'atol':tol}

    # Solve using different methods
    sol45 = solve_ivp (rhs_a,xd,y0, method='RK45',args=(E_d,),**options)
    sol23 = solve_ivp (rhs_a,xd,y0, method='RK23',args=(E_d,),**options)
    sol_Radau = solve_ivp (rhs_a,xd,y0, method='Radau',args=(E_d,),**options)
    sol_BDF = solve_ivp (rhs_a,xd,y0, method='BDF',args=(E_d,),**options)
    
    #calculate averge time steps
    step_sizes_rk45.append(np.mean(np.diff(sol45.t)))
    step_sizes_rk23.append(np.mean(np.diff(sol23.t)))
    step_sizes_radau.append(np.mean(np.diff(sol_Radau.t)))
    step_sizes_bdf.append(np.mean(np.diff(sol_BDF.t)))

#perform linear regression (log-log) to determine slopes
fit45 = np.polyfit(np.log(step_sizes_rk45), np.log(tols), 1)
fit23 = np.polyfit(np.log(step_sizes_rk23), np.log(tols), 1)
fit_radau = np.polyfit(np.log(step_sizes_radau), np.log(tols), 1)
fit_bdf = np.polyfit(np.log(step_sizes_bdf), np.log(tols), 1)

#extract slopes
A9 = [fit45[0], fit23[0], fit_radau[0], fit_bdf[0]]
#plot Part D
plt.figure(figsize=(10, 6))
plt.loglog(tols, step_sizes_rk45, label='RK45', marker='o')
plt.loglog(tols, step_sizes_rk23, label='RK23', marker='s')
plt.loglog(tols, step_sizes_radau, label='Radau', marker='^')
plt.loglog(tols, step_sizes_bdf, label='BDF', marker='x')

plt.xlabel('Tolerance')
plt.ylabel('Average Step Size')
plt.legend()
plt.grid(True)
plt.title('Convergence Study: Step Size vs Tolerance')
plt.show()

#Part E=======================================================

h = np.array([np.ones_like(xspan), 2*xspan, 4*(xspan**2) - 2, 8*(xspan**3) - 12*xspan, 16*xspan**4 - 48*xspan**2 + 12])

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Calculate the theoretical eigenfunctions `phi`
phi = np.zeros((len(xspan), 5))
for j in range(5):
    phi[:, j] = np.exp(-(xspan**2) / 2) * h[j, :] / np.sqrt(factorial(j) * (2**j) * np.sqrt(np.pi))

# Compute errors
erpsi_a = np.zeros(5)
erpsi_b = np.zeros(5)
er_a = np.zeros(5)
er_b = np.zeros(5)

for j in range(5):
    erpsi_a[j] = np.trapezoid((np.abs(ysola[:, j]) - np.abs(phi[:, j]))**2, xspan)  # Error for A1
    erpsi_b[j] = np.trapezoid((np.abs(ysolb[:, j]) - np.abs(phi[:, j]))**2, xspan)  
    
    er_a[j] = 100 * (np.abs(Esola[j] - (2*(j+1)-1)) / (2*(j+1)-1)) # Error for eigenvalues
    er_b[j] = 100 * (np.abs(Esolb[j] - (2*(j+1)-1)) / (2*(j+1)-1))

A10 = erpsi_a
A12 = erpsi_b
A11 = er_a
A13 = er_b

print(A13)
