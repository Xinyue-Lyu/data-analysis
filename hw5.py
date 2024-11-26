import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.linalg import lu, solve_triangular
import time
from matplotlib.animation import FuncAnimation

# Define parameters
tspan = np.arange(0, 4.5, 0.5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

#Initial conditions
w = np.exp(-X**2 - Y**2 / 20)
w2 = w.reshape(N)

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

#A, B, C coeficient matrix
m = 64
n = m * m
dx = 20/m

e0 = np.zeros((n, 1)) # vector of zeros
e1 = np.ones((n, 1)) # vector of ones
e2 = np.copy(e1) # copy the one vector
e4 = np.copy(e0) # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0 # overwrite every m^th value with zero
    e4[m*j-1] = 1 # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements for A
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(),
        e2.flatten(), -4 * e1.flatten(), e3.flatten(),
        e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]
matA = spdiags(diagonals, offsets, n, n).toarray()
A = matA / (dx**2)

A[0,0] = 2/(dx**2)

# Place diagonal elements for B
diagonals_B = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets_B = [-(n-m), -m, m, (n-m)]
matB = spdiags(diagonals_B, offsets_B, n, n).toarray() 
B = matB/ (2 * dx)

#plt.spy(matB)
#plt.show()

# Place diagonal elements for C
diagonals_c = [e5.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_c = [-m+1, -1, 1, m-1]
matC = spdiags(diagonals_c, offsets_c, n, n).toarray() 
C = matC / (dx*2)

# Define the ODE system
def fft_rhs(t, wt2, nx, ny, N, A, B, C, K, nu):
    # Reshape the flat array back to 2D
    wt = wt2.reshape((nx, ny))
    
    # Compute the streamfunction in Fourier space
    psit = -fft2(wt) / K
    psi = np.real(ifft2(psit)).reshape(N)  # Inverse FFT to get psi in real space
    
    # Compute the right-hand side of the vorticity equation
    rhs = nu * np.dot(A,wt2)+(np.dot(B,wt2))*(np.dot(C,psi))-(np.dot(B,psi))*(np.dot(C,wt2))
    return rhs

# Solve the ODE system
start_fft = time.time()
w_sol = solve_ivp( fft_rhs, [0, 4], w2, t_eval=tspan, 
                  args=(nx, ny, N, A, B, C, K, nu), method='RK45')
end_fft = time.time() 
A1 = w_sol.y
#print(f'A1: {A1}')
#print(f'FFT time = {end_fft - start_fft} seconds')

'''
for j, t in enumerate(tspan):
    w = A1[:, j].reshape((nx, ny))  # Reshape flat solution back to 2D
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w, shading='auto', cmap='jet')
    plt.title(f'Time: {t:.1f}')
    plt.colorbar()
'''

###################################PART B####################################

# 1.  Direct Solver (A\b)
def GE_rhs(t,w2, nx, ny, N, A, B, C, K, nu):
    psi= np.linalg.solve(A,w2)
    rhs=nu*np.dot(A,w2)+(np.dot(B,w2))*(np.dot(C,psi))-(np.dot(B,psi))*(np.dot(C,w2))
    return rhs

start_GE = time.time()
wtsol_GE = solve_ivp(GE_rhs, [0, 4], w2, t_eval=tspan, args=(nx, ny, N, A, B, C, K, nu), method='RK45')
end_GE = time.time() 
A2 = wtsol_GE.y
#print(f'A2: {A2}')
#print(f'A \ b time = {end_GE - start_GE} seconds')


# 2. LU Decomposition
start_LU = time.time()
P, L, U = lu(A)
def LU_rhs(t,w2, nx, ny, N, A, B, C, K,nu):
    Pb = np.dot(P, w2)
    y = solve_triangular(L, Pb, lower=True)
    psi = solve_triangular(U, y)
    rhs=nu*np.dot(A,w2)+(np.dot(B,w2))*(np.dot(C,psi))-(np.dot(B,psi))*(np.dot(C,w2))
    return rhs

wtsol_LU = solve_ivp(LU_rhs, [0, 4], w2, t_eval=tspan, args=(nx, ny, N, A, B, C, K, nu), method='RK45')
A3 = wtsol_LU.y
end_LU = time.time() 

#print(f'A3 : {A3}')
#print(f'LU time = {end_LU - start_LU} seconds')
 

#############################PART C##################################

# 1. Two oppositely “charged” Gaussian vorticies next to each other
w = np.exp(-((X + 5)**2 + (Y + 5)**2) / 20) - np.exp(-((X - 5)**2 + (Y - 5)**2) / 20)
w2 = w.reshape(N)
w_opp = solve_ivp( fft_rhs, [0, 4], w2, t_eval=tspan, 
                  args=(nx, ny, N, A, B, C, K, nu), method='RK45')
opp_sol = w_opp.y

# 2. Two same “charged” Gaussian vorticies next to each other.
w_same = np.exp(-((X + 5)**2 + (Y + 5)**2) / 20) + np.exp(-((X - 5)**2 + (Y - 5)**2) / 20)
w2_same = w_same.reshape(N)
w_same = solve_ivp( fft_rhs, [0, 4], w2_same, t_eval=tspan, 
                  args=(nx, ny, N, A, B, C, K, nu), method='RK45')
same_sol = w_same.y

# 3. Two pairs of oppositely “charged” vorticies
w_pairs = (
    np.exp(-((X - 5)**2 + (Y - 5)**2) / 20) - np.exp(-((X - 6)**2 + (Y - 5)**2) / 20) +  # Pair 1
    np.exp(-((X + 5)**2 + (Y + 5)**2) / 20) - np.exp(-((X + 6)**2 + (Y + 5)**2) / 20)    # Pair 2
)

# Flatten the vorticity field for solving
w2_pairs = w_pairs.reshape(N)

# Solve the dynamics for the two pairs of vortices
w_pairs_sol = solve_ivp(fft_rhs, [0, 4], w2_pairs, t_eval=tspan, 
    args=(nx, ny, N, A, B, C, K, nu), method='RK45')

# Extract the solution
pairs_sol = w_pairs_sol.y

# 4. random vorticities
num_vortices = 15
np.random.seed(42)  # Ensure reproducibility
w_rand = np.zeros_like(X)
for _ in range(num_vortices):
    x0, y0 = np.random.uniform(-10, 10, 2)
    strength = np.random.uniform(-1, 1)
    ellipticity = np.random.uniform(1, 5)
    w_rand += strength * np.exp(-((X - x0)**2 + ellipticity * (Y - y0)**2) / 20)
w2_rand = w_rand.reshape(N)
w_rand = solve_ivp( fft_rhs, [0, 4], w2_rand, t_eval=tspan, 
                  args=(nx, ny, N, A, B, C, K, nu), method='RK45')
rand_sol = w_rand.y

#visualization
fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
titles = ["Opposite Charge Vortices", "Same Charge Vortices", "Random Vortices", "Colliding Vortex Pairs"]

# Create pcolor plots for each subplot
cplots = []
for ax, title in zip(axs, titles):
    cplot = ax.pcolor(X, Y, np.zeros_like(X), shading='auto', cmap='jet', vmin=-1, vmax=1)
    ax.set_title(title)
    fig.colorbar(cplot, ax=ax)
    cplots.append(cplot)

# Update function for animation
def update(frame):
    cplots[0].set_array(opp_sol[:, frame].reshape((nx, ny)).flatten())
    cplots[1].set_array(same_sol[:, frame].reshape((nx, ny)).flatten())
    cplots[2].set_array(rand_sol[:, frame].reshape((nx, ny)).flatten())
    cplots[3].set_array(pairs_sol[:, frame].reshape((nx, ny)).flatten())
    return cplots

# Create the animation
anim = FuncAnimation(fig, update, frames=len(tspan), interval=200)
# Save the animation as an MP4 file using ffmpeg
anim.save('vorticity_dynamics.mp4', writer='ffmpeg', fps=10)

plt.tight_layout()
plt.show()
