import numpy as np

def computeCost(X, y, theta):
    m = y.size
    diff = np.dot(X, theta).reshape(-1) - y
    return np.vdot(diff, diff) / (2 * m)

def gradientDescent(X, y, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)
    theta = np.zeros(X.shape[1]) # initialize fitting parameters

    for idx in xrange(num_iters): # np.nditer(J_history, op_flags=['readwrite']):
        diff = (np.dot(X, theta).reshape(-1) - y).reshape((m,1))
        theta = theta - (alpha/m) * np.dot(X.T, diff).reshape(-1) # np.dot(diff.T, X).T
        
        # Save the cost J in every iteration    
        J_history[idx] = computeCost(X, y, theta)
        
    return theta, J_history

def featureNormalize(X_in):
    mean = np.mean(X_in, axis=0)
    mu = np.tile(mean, (X_in.shape[0], 1))
    sigma = np.tile(np.std(X_in, axis=0, ddof=1), (X_in.shape[0], 1))
    
    X_norm = X_in - mu
    X_norm = X_norm / sigma
    
    return (X_norm, mu[0, :], sigma[0, :])

def normalEqn(X, y):
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    # MATLAB: theta = pinv(X' * X) * X' * y;
    return theta
