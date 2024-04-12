import numpy as np
import matplotlib.pyplot as plt

import spatialmath.base as smb
from mrekf.transforms import pol2cart, cart2pol, inverse, forward

if __name__=="__main__":
    T0 = smb.transl2(0., 0.)
    # smb.trplot2(T0, frame="0", color="k")

    # True frames - blue
    # k0 - dashed lines and open circles
    # k1 - full lines and closed circles
    
    # True frame
    Tk1_t = smb.transl2(1.0931, -3.5345) @  smb.trot2(-4.6312)
    smb.trplot2(Tk1_t, frame="k1_t", color="b", width=0.2)
    Tk0_t = smb.transl2(1.1026, -3.634) @  smb.trot2(-4.617)
    smb.trplot2(Tk1_t, frame="k0_t", color="b", width=0.2, d1=0.1, d2=0.8)
    
    # Estimated frame
    Tk1_e = smb.transl2(0.9243, -3.8099) @  smb.trot2(1.5767)
    Tk0_e = smb.transl2(0.9488, -3.7512) @  smb.trot2(1.556)
    smb.trplot2(Tk1_e, frame="k1_e", color="r", width=0.2)
    smb.trplot2(Tk0_e, frame="ko_e", color="r", width=0.2, d1=0.1, d2=0.8)
    
    # Point
    # k 1
    p_k1_t = np.array([7.8881, -6.3533])
    smb.plot_point(p_k1_t, "ys", text="pk1_t")

    p_k1_e = np.array([8.4792, -6.3827])
    smb.plot_point(p_k1_e, "ys", text="pk1_e")
 
    # k 0 
    p_k0_t = np.array([7.820, -6.426])
    smb.plot_point(p_k0_t, "go", text="pk0_t")
    
    p_k0_e = np.array([7.0353, -8.0568])
    smb.plot_point(p_k0_e, "go", text="pk0_e")

    # FML
    p_k1_tf = forward(Tk1_e, np.array([-4.3443, -6.0608])) 
    smb.plot_point(p_k1_tf, "bo", text="pk1_tf")

    # Arrows
    # smb.plot_arrow(Tk1_e[:2,2], p_k1_e, linestyle=":", linewidth=0.5)
    # smb.plot_arrow(Tk0_e[:2,2], p_k0_e, linestyle=":", linewidth=0.5)

    # observations
    z = np.array([7.98, -1.8384])
    xy_z_k1e = pol2cart(z)
    xy_z_k0e = forward(Tk1_e, xy_z_k1e)
    xy_z_k0t = forward(Tk1_t, xy_z_k1e)
    smb.plot_arrow(Tk1_e[:2,2], xy_z_k0e, linestyle="-", linewidth=0.8)
    smb.plot_arrow(Tk1_t[:2,2], xy_z_k0t, linestyle=":", linewidth=0.8)

    

    plt.legend()
    plt.show()

    