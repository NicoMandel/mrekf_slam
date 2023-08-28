import numpy as np
from scipy.linalg import block_diag
from roboticstoolbox.mobile import VehicleBase

### section with static methods - pure mathematics, just gets used by every instance
def predict(x_est : np.ndarray, P_est : np.ndarray, robot : VehicleBase, odo, Fx : np.ndarray, Fv : np.ndarray, V : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
        TODO - change this, remove the indexing and f stuff!
        prediction function just mathematically
        requires the correct Fx and Fv dimensions
        V is the noise variance matrix. Is constructed by inserting the correct V in the right places.
        which means that for every dynamic landmark, a V must be inserted at the diagonal block. 
    """
    # state prediction - only predict the vehicle in the non-kinematic case. 
    xv_est = x_est[:3]
    xm_est = x_est[3:]
    xm_pred = xm_est
    xv_pred = robot.f(xv_est, odo)
    x_pred = np.r_[xv_pred, xm_pred]
    P_est = P_est
    
    # Covariance prediction
    # differs slightly from PC. see
    # [[/home/mandel/mambaforge/envs/corke/lib/python3.10/site-packages/roboticstoolbox/mobile/EKF.py]]
    # L 806
    P_pred = Fx @ P_est @ Fx.T + Fv @ V @ Fv.T

    return x_pred, P_pred

### update steps
def calculate_S(P_pred : np.ndarray, Hx : np.ndarray, Hw : np.ndarray, W_est : np.ndarray) -> np.ndarray:
    """
        function to calculate S
    """
    S = Hx @ P_pred @ Hx.T + Hw @ W_est @ Hw.T
    return S

def calculate_K(Hx : np.ndarray, S : np.ndarray, P_pred : np.ndarray) -> np.ndarray:
    """
        Function to calculate K
    """
    K = P_pred @ Hx.T @ np.linalg.inv(S)
    return K

def update_state(x_pred : np.ndarray, K : np.ndarray, innov : np.ndarray) -> np.ndarray:
    """
        Function to update the predicted state with the Kalman gain and the innovation
        K or innovation for not measured landmarks should both be 0
    """
    x_est = x_pred + K @ innov
    return x_est

def update_covariance_joseph(P_pred : np.ndarray, K : np.ndarray, W_est : np.ndarray, Hx : np.ndarray) -> np.ndarray:
    """
        Function to update the covariance
    """
    I = np.eye(P_pred.shape[0])
    P_est = (I - K @ Hx) @ P_pred @ (I - K @ Hx).T + K @ W_est @ K.T
    return P_est

def update_covariance_normal(P_pred : np.ndarray, S : np.ndarray, K : np.ndarray) -> np.ndarray:
    """
        update covariance
    """
    P_est = P_pred - K @ S @ K.T
    # symmetry enforcement
    P_est = 0.5 * (P_est + P_est.T)
    return P_est

### inserting new variables
def extend_map(x : np.ndarray, P : np.ndarray, xf : np.ndarray, Gz : np.ndarray, Gx : np.ndarray, W_est : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # extend the state vector with the new features
    x_ext = np.r_[x, xf]

    # extend the map
    n_x = len(x)
    n_lm = len(xf)

    Yz = np.block([
        [np.eye(n_x), np.zeros((n_x, n_lm))    ],
        [Gx,        np.zeros((n_lm, n_x-3)), Gz]
    ])
    P_ext = Yz @ block_diag(P, W_est) @ Yz.T

    return x_ext, P_ext