import numpy as np

# Given data matrix M
M = np.array([
    [12, 3, 6],
    [17, 13, -2],
    [12, 13, 3],
    [6, 13.5, -2.5],
    [17, 21, 7],
    [4, 20.5, -1]
])

# Number of rows (n)
n = M.shape[0]

# Create identity matrix I_n
I_n = np.identity(n)

# Create ones matrix J_n
J_n = np.ones((n, n))

# Calculate centering matrix C_n
C_n = I_n - 1/n * J_n

# Compute centered matrix MC = C_n * M
MC = np.dot(C_n, M)

print("Centered Matrix MC:")
print(MC)


NIn = I_n / n
print("\nReduced Identity Matrix NIn:")
print(NIn)

VarM = np.dot(np.dot(MC.T, NIn), MC)
print("\nVariance Matrix VarM:")
print(VarM)

VarV = np.diag(np.diag(VarM))
print("\nDiagonal Variance Matrix VarV:")
print(VarV)

IVarV = np.linalg.inv(VarV)
print("\nInverse Matrix IVarV:")
print(IVarV)

Mr = np.dot(MC, IVarV)
print("\nReduced Matrix Mr:")
print(Mr)

Cov = np.dot(np.dot(Mr.T, NIn), Mr)
print("\nCovariance Matrix Cov:")
print(Cov)