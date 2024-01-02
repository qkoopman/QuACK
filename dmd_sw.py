import numpy as np
import time

def sliding_window_dmd_nonsq(
        params, window_size: int, start_time: int, end_time: int, pred_time_len: int,
):
    """
        Sliding-Window Dynamic Mode Decomposition (SW-DMD) that adopts a generally nonsquare Koopman-like matrix K.

            X_prime ~ K @ X

            Optimal K = X_prime @ pseudo_inverse(X)

            K: (generally nonsquare) Koopman-like operator; shape (num_params, num_params * window_size)
            X: data matrix of the previous time (previous-time data) with sliding-window embedding; shape: (num_params * window_size, end_time - start_time)
            X_prime: data matrix of the the current time (without sliding-window embedding); shape: (num_params, end_time - start_time)

            At window_size = 1, the SW-DMD is reduced to the standard DMD.

        ## Args:
            params (numpy array):
                shape (num_iters, num_params); The original data matrix (parameters from opimization history) before sliding-window embedding; Axis 0 is for optimization iterations; Aixs 1 is for components of parameters;
            window_size (int):
                the size of the sliding window;
            start_time (int):
                (inclusive) the starting time index of data used in X
            end_time (int):
                (exclusive) the end time index of data used in X; This is also equal to the (inclusive) last time index of data used in X_prime (because times used in X_prime is one step more forward than those in X).
            pred_time_len (int):
                the length of time for predicting data (in the future optimization);
        
        ## Returns:
            X_pred (numpy array):
                shape: (num_params, pred_time_len); predicted parameters;

    """
    
    start_run_time = time.time()

    params_in_window = []
    num_iters, num_params = params.shape
    for t in range(num_iters - window_size + 1):
        params_in_window.append(np.concatenate(([params[t + _, :] for _ in range(window_size)])))
    params_in_window = np.array(params_in_window).transpose() # shape: (num_params * window_size, num_iters - window_size + 1)

    X = params_in_window[:, start_time:end_time] # shape: (num_params * window_size, end_time - start_time)
    X_prime = params.transpose()[:, start_time + window_size:end_time + window_size] # shape: (num_params, end_time - start_time)

    K = X_prime @ np.linalg.pinv(X) # shape: (num_params, num_params * window_size)

    X_last = params_in_window[:, end_time].reshape((-1, 1)) # the last time index, i.e., end_time, used in X_prime (which is absent in X)
    X_pred = [K @ X_last]

    for t in range(end_time, end_time + pred_time_len - 1):
        X_last = np.concatenate((X_last[num_params:, :], X_pred[-1]), axis=0)
        X_pred.append(K @ X_last)
    # len(X_pred) (as a list): pred_time_len
    X_pred = np.array(X_pred) # shape: (pred_time_len, num_params)
    X_pred = X_pred.reshape((X_pred.shape[0], X_pred.shape[1])).transpose() # shape: (num_params, pred_time_len)

    print("Time elapsed (seconds):", time.time() - start_run_time)
    return X_pred