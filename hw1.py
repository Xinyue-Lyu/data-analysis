import numpy as np

#Part 1
def f(x):
    return x*np.sin(3*x)-np.exp(x)

def f_prime(x):
    return np.sin(3*x) + 3*x*np.cos(3*x) - np.exp(x)

#(i)
A1 = np.array([-1.6])
A3 = [[0], [0]]
for j in range(100):
    f_new = f(A1[j])
    A1 = np.append(A1, A1[j] - f(A1[j])/f_prime(A1[j]))
    if abs(f_new) < 10**(-6):
        A3[0] = j + 1
        break

#(ii)
xr = -0.4
xl = -0.7
A2 = []

for j in range(0,100):
    xc = (xr+xl)/2
    A2.append(xc)
    fc = f(xc)
    if (f(xr)*f(xc)<0):
        xl = xc
    else:
        xr = xc
    if(abs(fc) < 10**(-6)):
        A3[1] = j + 1
        break
print(A2)

#Part 2
A = np.array([[1,2], [-1,1]])
B = np.array([[2,0], [0,2]])
C = np.array([[2,0, -3], [0, 0, -1] ])
D = np.array([ [1, 2], [2, 3], [-1, 0]])
x = np.array([1, 0])
y = np.array([0, 1])
z = np.array([1, 2, -1])

A4 = A+B
 
A5 = 3*x-4*y

A6 = A@x

A7 = B@(x-y)

A8 = D@x

A9 = D@y + z

A10 = A@B

A11 = B@C

A12 = C@D

