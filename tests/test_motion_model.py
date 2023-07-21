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
    