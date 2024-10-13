import numpy as np

def generate_samples(q_as, q_bs, sz_sample=50, n_exp=100):
    indices = np.random.choice(q_as.shape[0], size=sz_sample*n_exp, replace=False)
    samples = []
    for i in range(n_exp):
        start, end = i*sz_sample, (i+1)*sz_sample
        samples.append([q_as[indices[start:end], :], q_bs[indices[start:end], :]])
    return samples

def calculate_error(R, t, R_hat, t_hat):
    e_R = np.arccos( (np.trace(R_hat.T @ R) - 1) / 2 )
    e_T = np.arccos(t.T @ t_hat)
    return e_R, e_T