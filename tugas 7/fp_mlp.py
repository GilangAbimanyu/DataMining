def forwardPass(inputs, weight, bias, activation = 'linear'):
	w_sum = np.dot(inputs, weight) + bias

	if activation is 'relu' :
		# ReLU Activation f(x) = max(0, x)
		act = np.maximum(w_sum, 0)
	else :
		# Linear Activation f(x) = x
		act = w_sum

	return act