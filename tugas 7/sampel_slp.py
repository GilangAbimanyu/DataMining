import numpy as np

def forwardPass(inputs, weight, bias):
	w_sum = np.dot(inputs, weight) + bias

	# Linear Activation f(x) = x
	act = w_sum

	return act

# Pre-Trained Weights & Biases after Training
W = np.array([[2.99999928]])
b = np.array([1.99999976])

# Initialize Input Data
inputs = np.array([[7], [8], [9], [10]])

# Output of Output Layer
o_out = forwardPass(inputs, W, b)

print('Output Layer Output (Linear)')
print('================================')
print(o_out, "\n")

"""
[[ 22.99999472]
 [ 25.999994  ]
 [ 28.99999328]
 [ 31.99999256]]
"""