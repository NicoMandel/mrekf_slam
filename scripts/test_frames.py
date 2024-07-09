import os.path
import numpy as np
import matplotlib.pyplot as plt

import spatialmath.base as smb
from mrekf.transforms import pol2cart, cart2pol, inverse, forward

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.transforms as tfs

# TODO - plot this in three different frames
# TODO - ask ChatGPT to create an icon from a clearpath Jackal image
# 0 - initial step
# 1 update, reframe, observation
# 2 - new frame -> ensure that this is 
# keep some of the last ones in there. 
# adjust line thickness

def getIcon(path : str):
    return OffsetImage(plt.imread(path), zoom=.05)

if __name__=="__main__":
    T0 = smb.transl2(0., 0.)
    # smb.trplot2(T0, frame="0", color="k")

    # True frames - blue
    # k0 - dashed lines and open circles
    # k1 - full lines and closed circles
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    carpath = os.path.join(basedir, 'results', 'Figures', 'car_icon.png')
    caricon = getIcon(carpath)

    # True frame
    fig, axs = plt.subplots(1,3)

    # time k0:
    Tk0_t = smb.transl2(1.1026, -3.634) @  smb.trot2(-4.617)
    smb.trplot2(Tk0_t, ax=axs[0], frame="k0_t", color="b", width=0.2, d1=0.1, d2=0.8)
    # tft = axs[0].transData + tfs.Affine2D().rotate_deg(-45)
    # ab = AnnotationBbox(caricon, (1.1026, -3.634), frameon=False)
    # ab.set_transform(tft)
    # axs[0].add_artist(ab)
    # estimated frame
    Tk0_e = smb.transl2(0.9488, -3.7512) @  smb.trot2(1.556)
    smb.trplot2(Tk0_e, ax=axs[0], frame="ko_e", color="r", width=0.2, d1=0.1, d2=0.8)
    # points
    p_k0_t = np.array([7.820, -6.426])
    smb.plot_point(p_k0_t,  "go", ax=axs[0], text="pk0_t")
    # estimated
    p_k0_e = np.array([7.0353, -8.0568])
    smb.plot_point(p_k0_e, "go", ax=axs[0], text="pk0_e")

    smb.trplot2(Tk0_e, ax=axs[1], frame="ko_e", color="r", width=0.2, d1=0.1, d2=0.8)

    # time k1: true and estimated
    Tk1_t = smb.transl2(1.0931, -3.5345) @  smb.trot2(-4.6312)
    smb.trplot2(Tk1_t, ax=axs[2], frame="k1_t", color="b", width=0.2)
    # Estimated frame    
    Tk1_e = smb.transl2(0.9243, -3.8099) @  smb.trot2(1.5767)
    smb.trplot2(Tk1_e, ax=axs[2], frame="k1_e", color="r", width=0.2)
    
    # Point
    # k 1
    p_k1_t = np.array([7.8881, -6.3533])
    smb.plot_point(p_k1_t, "ys", ax=axs[2], text="pk1_t")
    smb.plot_point(p_k1_t, "ys", ax=axs[1], text="pk1_t")


    p_k1_e = np.array([8.4792, -6.3827])
    smb.plot_point(p_k1_e,  "ys", ax=axs[2], text="pk1_e")
 
    # FML
    p_k1_tf = forward(Tk1_e, np.array([-4.3443, -6.0608])) 
    smb.plot_point(p_k1_tf, "bo", ax=axs[1], text="pk1_tf")

    # Arrows
    # smb.plot_arrow(Tk1_e[:2,2], p_k1_e, linestyle=":", linewidth=0.5)
    # smb.plot_arrow(Tk0_e[:2,2], p_k0_e, linestyle=":", linewidth=0.5)

    # observations
    z = np.array([7.98, -1.8384])
    xy_z_k1e = pol2cart(z)
    xy_z_k0e = forward(Tk1_e, xy_z_k1e)
    xy_z_k0t = forward(Tk1_t, xy_z_k1e)
    smb.plot_arrow(Tk1_e[:2,2], xy_z_k0e, ax=axs[1], linestyle="-", linewidth=0.8)
    smb.plot_arrow(Tk1_t[:2,2], xy_z_k0t, ax=axs[0], linestyle=":", linewidth=0.8)

    axs[0].set_title("k=0 Predict")
    axs[1].set_title("k=0 Update")
    axs[2].set_title("k=1")
    axs[0].set_xlim(-1,9)
    axs[1].set_xlim(-1,9)
    axs[0].set_ylim(-9,-1)
    axs[1].set_ylim(-9,-1)
    axs[2].set_xlim(-1,9)
    axs[2].set_ylim(-9,-1)
    # plt.legend()
    plt.show()

    