import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp

beta = 1
D1 =0.1
D2 =0.1
T = 4
tspan = np.arange(0, 4.5, 0.5)
nu = 0.001
nx =64
ny =64
N = nx * ny
Lx = 20
Ly = 20

# Define spatial domain
x2 = np.linspace(-10, 10, nx + 1)
x = x2[:nx]
y2 = np.linspace(-10, 10, ny + 1)
y = y2[:ny]
[X,Y] = np.meshgrid(x,y);

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2
K[0, 0] = 1e-6

#compute u & v
m=1; # number of spirals
u = np. tanh(np.sqrt(x**2 + Y**2)) * np.cos(m * np.angle(X + 1j* Y) - np.sqrt (X**2 + Y**2) )
v = np. tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + 1j * Y) - np.sqrt (X**2 + Y**2))

ut = fft2(u)
vt = fft2(v)
uvt0 = np.hstack([(ut.reshape(N)),(vt.reshape(N))])

# define PDE
def rhs(t, uvt, nx, ny, N, beta, D1, D2, K):
    # split up uvt into utc & vtc
    utc = uvt[:N]
    vtc = uvt[N:]
    ut =utc.reshape((nx,ny))
    vt = vtc.reshape((nx,ny))
    u = ifft2(ut)
    v = ifft2(vt)

    A_2 = u**2 + v**2
    lam = 1 - A_2
    w = -beta*A_2

    rhs_u = (-D1*K*ut + fft2(lam*u - w*v)).reshape(N)
    rhs_v = (-D2 * K * vt + fft2(w*u + lam*v)). reshape(N)
    rhs = np.hstack([(rhs_u), (rhs_v)])

    return rhs

uvtsol = solve_ivp(rhs, [0,T], uvt0, t_eval = tspan, args=(nx, ny, N, beta, D1, D2, K))
z = uvtsol.y
A1 = np.real(uvtsol.y)
#print(A1[1,0])

'''
#plot
plt.figure(figsize=(12, 12))
for j, t in enumerate(tspan):
	ut = z[:N, j].reshape((nx, ny))
	u = np.real(ifft2(ut))
	plt.subplot(3,3,j+1)
	plt.pcolor(x, y, u, cmap='RdBu_r')
	plt.colorbar()

plt.tight_layout()
plt.show()
'''
#######################PARTB############################
#define chebychev
def cheb(N):
	if N==0: 
		D = 0.; x = 1.
	else:
		n = np.arange(0,N+1)
		x = np.cos(np.pi*n/N).reshape(N+1,1) 
		c = (np.hstack(( [2.], np.ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
		X = np.tile(x,(1,N+1))
		dX = X - X.T
		D = np.dot(c,1./c.T)/(dX+np.eye(N+1))
		D -= np.diag(np.sum(D.T,axis=0))
	return D, x.reshape(N+1)

#define parameters
N = 30
D,x = cheb(N)
D[N,:] = 0
D[0,:] = 0
Dxx = np.dot(D,D)/ ((20/2)**2)

y = x 
N2 = (N+1) * (N+1)
I = np.eye(len(Dxx))
L = np.kron(I, Dxx) + np.kron(Dxx, I)
X,Y = np.meshgrid(x,y)
X = X*(20/2)
Y = Y*(20/2)

u = np. tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(X + 1j* Y) - np.sqrt (X**2 + Y**2) )
v = np. tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + 1j * Y) - np.sqrt (X**2 + Y**2))

uv0 = np.hstack([(u.reshape(N2)),(v.reshape(N2))])

#def function
def rhs2(t, uv, N2, beta, D1, D2):
    # split up uvt into utc & vtc
    u = uv[:N2]
    v = uv[N2:]

    A_2 = u**2 + v**2
    lam = 1 - A_2
    w = -beta*A_2

    rhs_u = D1*np.dot(L,u) + lam * u - w * v
    rhs_v = D2*np.dot(L,v) + w * u + lam * v
    rhsc = np.hstack([rhs_u, rhs_v])
    return rhsc

uvsol = solve_ivp(rhs2, [0,T], uv0, t_eval = tspan,method='RK45', args=(N2, beta, D1, D2))
A2 = uvsol.y
print(A2)