import numpy as np

def eight_point_algorithm(q_a, q_b):
    x_a, y_a = q_a[:, 0].reshape(-1, 1), q_a[:, 1].reshape(-1, 1)
    x_b, y_b = q_b[:, 0].reshape(-1, 1), q_b[:, 1].reshape(-1, 1)
    M = np.hstack((
        x_b*x_a,
        x_b*y_a,
        x_b,
        y_b*x_a,
        y_b*y_a,
        y_b,
        x_a,
        y_a,
        np.ones(shape=(q_a.shape[0], 1))
    ))
    u, d, vT = np.linalg.svd(M)
    return vT[-1, :].reshape(3, 3)

def decompose_essential_matrix(E):
    u, d, vT = np.linalg.svd(E)

    if np.isclose(np.linalg.det(u), -1):
        u = -u
    if np.isclose(np.linalg.det(vT), -1):
        vT = -vT

    d = np.diag([1, 1, 0])

    return u, d, vT

def find_4d_point(q_ia, q_ib, P_ia, P_ib):
    x_ia, y_ia = q_ia[0], q_ia[1]
    x_ib, y_ib = q_ib[0], q_ib[1]
    A = np.array([
        y_ia * P_ia[2, :] - P_ia[1, :],
        x_ia * P_ia[2, :] - P_ia[0, :],
        y_ib * P_ib[2, :] - P_ib[1, :],
        x_ib * P_ib[2, :] - P_ib[0, :],
    ])
    _, _, vT = np.linalg.svd(A)
    return vT[-1, :]

def find_rotation_and_translation2(q_a, q_b, u, d_, vT):
    t = u[:, -1].reshape(-1, 1)

    mid_m_a = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])
    mid_m_b = np.array([
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0, 0, 1]
    ])

    R_a = u @ mid_m_a @ vT
    R_b = u @ mid_m_b @ vT

    l = [(R_a, t), (R_a, -t), (R_b, t), (R_b, -t)]

    P_ia = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    correct = {}
    make_h = lambda R, t : np.vstack((np.hstack((R, t))))

    for i, (R_ib, t_ib) in enumerate(l):
        P_ib = make_h(R_ib, t_ib)

        num_true = 0
    
        for (q_ia, q_ib) in zip(q_a, q_b):
            Q_ia = find_4d_point(q_ia, q_ib, P_ia, P_ib)

            q_ia_E = Q_ia/Q_ia[-1] 
            q_ib_E = (P_ib @ Q_ia).flatten()

            num_true += q_ia_E[2] * q_ia_E[3] > 0 and q_ib_E[2] > 0

        correct[i] = num_true
    
    i = max(correct, key=correct.get) 
    R_b_final, t_b_final = l[i]
    return R_b_final, t_b_final

def hartley_normalization(q_a, q_b):
    q_a = np.append(q_a, np.ones(shape=(q_a.shape[0], 1)),axis=1).reshape(-1, 3)
    q_b = np.append(q_b, np.ones(shape=(q_b.shape[0], 1)),axis=1).reshape(-1, 3)

    x_a = q_a[:, 0]
    y_a = q_a[:, 1]
    x_b = q_b[:, 0]
    y_b = q_b[:, 1]

    m_a = np.mean(q_a, axis=0)
    m_b = np.mean(q_b, axis=0)

    s_a = (1/(2*m_a.shape[0]) * np.sum((x_a - m_a[0])**2 + (y_a - m_a[1])**2))**0.5
    s_b = (1/(2*m_b.shape[0]) * np.sum((x_b - m_b[0])**2 + (y_b - m_b[1])**2))**0.5

    T_a = np.array([
        [1/s_a, 0, -m_a[0]/s_a],
        [0, 1/s_a, -m_a[1]/s_a],
        [0, 0, 1]
    ])
    
    T_b = np.array([
        [1/s_b, 0, -m_b[0]/s_b],
        [0, 1/s_b, -m_b[1]/s_b],
        [0, 0, 1]
    ])

    q_a_tr = q_a @ T_a
    q_b_tr = q_b @ T_b

    return q_a_tr[:, :-1], q_b_tr[:, :-1], T_a, T_b


def recover_actual_essential_matrix(F, T_a, T_b):
    return np.linalg.inv(T_b.T) @ F @ np.linalg.inv(T_a)