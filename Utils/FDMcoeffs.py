# Compute finite difference coefficients
import numpy as np
import math as m

# n for stencils in fu
# k for start index, e.g. CD4: (-2, -1, 0, 1, 2) -> coeff(5, -2, 1)
def coeff(N, k, order):
    base = np.ones(N)
    for i in range(N):
        base[i] = k+i
    b = np.zeros(N)
    b[order] = 1
    A = np.ones(N)
    for i in range(1, N):
        A = np.vstack([A,base**i/m.factorial(i)])
    coeff = np.linalg.solve(A, b)
    return coeff

def fluxCoeff(N, k):
    coef = coeff(N,k,1)
    # F(j+1/2) coeff
    F_coeff = np.zeros(N - 1)
    F_coeff[0] = -coef[0]
    for i in range(1,N-1):
        F_coeff[i] = F_coeff[i-1] - coef[i]
    return F_coeff

# Mixing CD6(0.3), UP5(0.7)
a = fluxCoeff(7, -3)
b = fluxCoeff(8, -4)
a = np.concatenate(([0], a))
# b = np.append(b, 0)
print(b*0.5+0.5*a)
