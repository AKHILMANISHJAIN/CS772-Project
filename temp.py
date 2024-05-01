import numpy as np

# Define the vector
vector = np.arange(1, 44000)

# Reshape the vector into a column vector
column_vector = vector.reshape(-1, 1)
print(vector.shape)
# Compute the pseudoinverse
pseudo_inverse = np.linalg.pinv(column_vector)

print("Pseudoinverse of the vector:")
print(pseudo_inverse)
