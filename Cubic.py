#==================================================================
#----------------------    Cubic method    ------------------------
#==================================================================
# A method of performing smoother interpolation by increasing 
# the expressive power of the auxiliary line by connecting the two 
# points with a cubic equation and by limiting the differential 
# information of the interpolation line at each interpolation point.
#==================================================================
# Pattern 1 : Basic implementation in Python
#==================================================================

import numpy as np
import matplotlib.pyplot as plt

def get_cubic_interpolate_function_coefficient(smpxy):
    
    N = len(smpxy) - 1
    
    X = np.zeros([4*N, 4*N])
    Y = np.zeros([4*N])
    
    ridx = 0
    
    #--------------- Equation start ---------------
    
    # Equation 1
    for i in range(N):
        X[ridx, i*4] = smpxy[i][0]**3
        X[ridx, i*4 + 1] = smpxy[i][0]**2
        X[ridx, i*4 + 2] = smpxy[i][0]
        X[ridx, i*4 + 3] = 1
        Y[ridx] = smpxy[i][1]
        
        ridx += 1
        
    # Equation 2
    for i in range(N):
        X[ridx, i*4] = smpxy[i + 1][0]**3
        X[ridx, i*4 + 1] = smpxy[i + 1][0]**2
        X[ridx, i*4 + 2] = smpxy[i + 1][0]
        X[ridx, i*4 + 3] = 1
        Y[ridx] = smpxy[i + 1][1]
        
        ridx += 1
        
    # Equation 3
    for i in range(N - 1):
        X[ridx, i*4] = 3*smpxy[i + 1][0]**2
        X[ridx, i*4 + 1] = 2*smpxy[i + 1][0]
        X[ridx, i*4 + 2] = 1
        # X[ridx, i*4 + 3] = 0
        
        X[ridx, (i + 1)*4] = -3*smpxy[i + 1][0]**2
        X[ridx, (i + 1)*4 + 1] = -2*smpxy[i + 1][0]
        X[ridx, (i + 1)*4 + 2] = -1
        # X[ridx, (i + 1)*4 + 3] = 0
        # Y[ridx] = 0
        
        ridx += 1
    
    # Equation 4
    for i in range(N - 1):
        X[ridx, i*4] = 3*smpxy[i + 1][0]
        X[ridx, i*4 + 1] = 1
        # X[ridx, i*4 + 2] = 0
        # X[ridx, i*4 + 3] = 0
        
        X[ridx, (i + 1)*4] = -3*smpxy[i + 1][0]
        X[ridx, (i + 1)*4 + 1] = -1
        # X[ridx, (i + 1)*4 + 2] = 0
        # X[ridx, (i + 1)*4 + 3] = 0
        # Y[ridx] = 0
        
        ridx += 1
        
    # Equation 5
    X[-2, 0] = 3*smpxy[0][0]
    X[-2, 1] = 1
    # X[-2, 2] = 0
    # X[-2, 3] = 0
    # Y[-2] = 0
    
    # Equation 6
    X[-1, -4] = 3*smpxy[-1][0]
    X[-1, -4 + 1] = 1
    # X[-1, -4 + 2] = 0
    # X[-1, -4 + 3] = 0
    # Y[-1] = 0
    
    #--------------- Equation end -----------------
    
    invX = np.linalg.inv(X)
    A = np.dot(invX, Y)
    
    return A

def cubic_inter_at_2point(idx, A):
    
    a = A[4*idx]
    b = A[4*idx + 1]
    c = A[4*idx + 2]
    d = A[4*idx + 3]
    
    def cubic_eq(x):
        return a*(x**3) + b*(x**2) + c*x + d
    
    return cubic_eq

def get_cubic_interpolate_function(smpx, smpy):
    
    smpxy = zip(smpx, smpy)
    smpxy = sorted(smpxy, key=lambda t: t[0])
    smpxy = list(smpxy)
    
    A = get_cubic_interpolate_function_coefficient(smpxy)
    
    def interpolate(intrx):
        if (intrx < smpxy[0][0]) | (intrx > smpxy[-1][0]):
            print("Interpolation point is not within sample point range.")
            return
        
        for idx in range(len(smpxy)):
            if intrx < smpxy[idx][0]:
                
                a = A[4*(idx - 1)]
                b = A[4*(idx - 1) + 1]
                c = A[4*(idx - 1) + 2]
                d = A[4*(idx - 1) + 3]
                return a*(intrx**3) + b*(intrx**2) + c*intrx + d
            
    return interpolate

x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2 / 10.0)
f = get_cubic_interpolate_function(x, y)

intrx = np.linspace(0, 10, num=101, endpoint=True)

plt.plot(x, y, 'o', intrx, [f(xi) for xi in intrx], '--')
plt.legend(['data', 'cubic'], loc='best')
plt.savefig("Cubic_by_Basic.png", dpi=300)
plt.show()

#==================================================================
# Pattern 2 : Cubic method using Scipy's API
#==================================================================

from scipy.interpolate import interp1d

x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2 / 10.0)

f = interp1d(x, y, kind='cubic')

intrx = np.linspace(0, 10, num=101, endpoint=True)

plt.plot(x, y, 'o', intrx, [f(xi) for xi in intrx], ':')
plt.legend(['data', 'cubic(scipy)'], loc='best')
plt.savefig("Cubic_by_scipy.png", dpi=300)
plt.show()

