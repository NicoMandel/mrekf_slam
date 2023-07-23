import numpy as np
from mrekf.motionmodels import StaticModel, KinematicModel

def init_sm(V = None) -> StaticModel:
    if V is None:
        V = np.eye(2)
    return StaticModel(V)

def init_kin(V = None, dt = 0.1) -> KinematicModel:
    if V is None:
        V = np.eye(4)
    return KinematicModel(V, dt)

def test_init():
    sm = init_sm()
    km = init_kin()

def get_off_diagonal(mat : np.ndarray):
    return mat - np.diag(np.diagonal(mat))

def test_properties():
    sm = init_sm()
    km = init_kin()
    assert sm.state_length == 2, "State length of static model not correct. Should be 2"
    assert km.state_length == 4, "State length of kinematic model not correct. Should be 4"
    assert sm.V.shape[0] == 2, "Covariance length of static model not correct. Should be 2"
    assert km.V.shape[0] == 4, "Covariance length of kinematic model not correct. Should be 4"
    assert km.A.shape[0] == 4, "Length of state transition matrix for kinematic model not correct. Should be 4"
    assert isinstance(km.dt, float), "Dt not of type float. Double check please" 

def test_methods():
    sm = init_sm()
    km = init_kin()
    x_s = np.array([1.2, 1.2])
    a = km.A
    x_k = np.array([1., 1.1, 1.2, 1.3])

    # state transition
    x_kk = a @ x_k
    assert np.allclose(x_kk, km.f(x_k)), "Prediction function f of kinematic model not correct. double check"
    assert np.allclose(x_s, sm.f(x_s)), "Prediction function f of static model not correct. double check"
    
    # Jacobians for static model
    dim = 2
    F_test = np.eye(dim)
    assert np.allclose(F_test, sm.Fx()), "Jacobian Fx of static model not correct. Double Check"
    assert np.allclose(F_test, sm.Fv()), "Jacobian Fv of static model not correct. Double Check"

    # Jacobians for kinematic model
    # Fx should be a diagonal
    Fx_k = km.Fx()
    nz = np.count_nonzero(get_off_diagonal(Fx_k))
    assert nz == 2, "Kinematic model Fx off-diagonal elements should be 2 (dt for x_dot and y_dot). Double Check"
    Fv_k = km.Fv()
    Fv_k1 = Fv_k[:,:2]
    Fv_k2 = Fv_k[:,2:]
    Fv_k21 = Fv_k2[:2,:]
    Fv_k22 = Fv_k2[2:,:] 
    assert np.allclose(np.zeros((4,2)), Fv_k1), "Kinematic Model Fv first two columns are not all zeros. Double Check"
    assert np.count_nonzero(get_off_diagonal(Fv_k21)) == 0, "Kinematic Model Fv (0,4) and (1,3) are not zeros. Double Check"
    assert np.count_nonzero(get_off_diagonal(Fv_k22)) == 0, "Kinematic Model Fv (2,4) and (3,3) are not zeros. Double Check"

def test_scale():
    V_stat = np.ones((2,2), dtype=float)
    V_kin = np.ones((4,4), dtype=float)
    sm = init_sm(V_stat)
    km = init_kin(V_kin)

    scale = 0.1
    sVs = sm.scale_V(scale)
    sVk = km.scale_V(scale)
    assert np.allclose(sVs, scale * V_stat), "Scaling function not working for Static model"
    assert np.allclose(sVk, scale * V_kin), "Scaling function not working for Kinematic model"
    