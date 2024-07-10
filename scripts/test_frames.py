import os.path
import numpy as np
import matplotlib.pyplot as plt

import spatialmath.base as smb
from mrekf.transforms import pol2cart, cart2pol, inverse, forward
from scipy.ndimage import rotate

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.transforms as tfs

# TODO - plot this in three different frames
# TODO - ask ChatGPT to create an icon from a clearpath Jackal image
# 0 - initial step
# 1 update, reframe, observation
# 2 - new frame -> ensure that this is 
# keep some of the last ones in there. 
# adjust line thickness

def icon_rotate(xytheta : np.ndarray, israd : bool = False, zoom : float = .1, path : str = None) -> AnnotationBbox:
    xyt = xytheta.copy()
    if path is None:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'Jackal_base.png'))
    if israd: xyt[2] = np.rad2deg(xyt[2])
    img = plt.imread(path)
    rotated_img = rotate(img, xyt[2] - 90, reshape=True)
    imgbox = OffsetImage(rotated_img, zoom=zoom)
    anbox = AnnotationBbox(imgbox, xyt[:2], frameon=False, boxcoords="data")
    return anbox

def getIcon(path : str) -> OffsetImage:
    return OffsetImage(plt.imread(path), zoom=.1)

if __name__=="__main__":
    # global settings
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    T0 = smb.transl2(0., 0.)
    xlim = (-1.5, 9)
    ylim = (-9,1.5)

    # k 0 -> true position and true and estimated frame
    fig, ax = plt.subplots()
    smb.trplot2(T0, ax=ax, frame="0", color="k", width=1.5, d2=1.2) #  d1=1.2

    # frames
    xytheta_t_0 = [1.1026, -3.634, -4.617]
    ab = icon_rotate(xytheta_t_0, israd=True)
    ax.add_artist(ab)
    Tk0_t = smb.transl2(xytheta_t_0[0], xytheta_t_0[1]) @  smb.trot2(xytheta_t_0[2])
    smb.trplot2(Tk0_t, ax=ax, frame=None, color="b", width=1.5, d2=1.1)         # width=0.2, d1=0.1, d2=0.8

    xytheta_e_0 = [0.9488, -3.7512, 1.556]
    Tk0_e = smb.transl2(xytheta_e_0[0], xytheta_e_0[1]) @  smb.trot2(xytheta_e_0[2])
    smb.trplot2(Tk0_e, ax=ax, frame=None, color="r", width=1.5, d2=1.1)

    # points
    p_k0_t = np.array([7.820, -6.426])
    smb.plot_point(p_k0_t,  "bo", ax=ax, text="p_t")
    # estimated
    p_k0_e = np.array([7.0353, -8.0568])
    smb.plot_point(p_k0_e, "ro", ax=ax, text="p_e")

    # ax.set_title("k=0 Predict")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.tight_layout()
    plt.show()

    # update - Estimate is the moment k_0
    fig, ax = plt.subplots()
    smb.trplot2(T0, ax=ax, frame="0", color="k", width=1.5, d2=1.2)

    ab = icon_rotate(xytheta_e_0, israd=True)
    ax.add_artist(ab)

    smb.trplot2(Tk0_e, ax=ax, frame=None, color="r", width=1.5, d2=1.1)
    smb.plot_point(p_k0_e, "ro", ax=ax, text="p_e")
    p_k1_tf = forward(Tk0_e, np.array([-4.3443, -6.0608])) 
    smb.plot_point(p_k1_tf, "go", ax=ax, text="p_tf", textargs={"horizontalalignment": "right","verticalalignment": "top"})
   
    # observations
    z = np.array([7.98, -1.8384])
    xy_z_k1e = pol2cart(z)
    xy_z_k0e = forward(Tk0_e, xy_z_k1e)
    # xy_z_k0t = forward(Tk1_t, xy_z_k1e)
    arr_kwargs ={"head_width" : 0.5}
    smb.plot_point(p_k0_t,  "bo", ax=ax, text="p_t")

    smb.plot_arrow(Tk0_e[:2,2], xy_z_k0e, ax=ax, linestyle="-", linewidth=1.5, **arr_kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.set_title("k=0 Update")
    plt.tight_layout()
    plt.show()

    # After update step
    fig, ax = plt.subplots()
    smb.trplot2(T0, ax=ax, frame="0", color="k", width=1.5, d2=1.2) #  d1=1.2

    xytheta_t_1 = [1.0931, -3.5345, -4.6312]
    xytheta_e_1 = [0.9243, -3.8099, 1.5767]
    # time k1: true and estimated
    Tk1_t = smb.transl2(xytheta_t_1[0], xytheta_t_1[1]) @  smb.trot2(xytheta_t_1[2])
    Tk1_e = smb.transl2(xytheta_e_1[0], xytheta_e_1[1]) @  smb.trot2(xytheta_e_1[2])
    ab = icon_rotate(xytheta_e_1, israd=True)
    ax.add_artist(ab)
    smb.trplot2(Tk1_t, ax=ax, frame=None, color="b", width=1.5, d2=1.1)  
    # Estimated frame
    smb.trplot2(Tk1_e, ax=ax, frame=None, color="r", width=1.5, d2=1.1)      
    # smb.trplot2(Tk1_e, ax=ax, frame="k1_e", color="r", width=0.2)
    
    # Point
    # k 1
    p_k1_t = np.array([7.8881, -6.3533])
    smb.plot_point(p_k1_t, "bo", ax=ax, text="p_t", textargs={"horizontalalignment": "right","verticalalignment": "top"})
    # smb.plot_point(p_k1_t, "ys", ax=ax, text="pk1_t")


    p_k1_e = np.array([8.4792, -6.3827])
    smb.plot_point(p_k1_e,  "ro", ax=ax, text="p_e")
 
    # FML
    # Arrows
    # smb.plot_arrow(Tk1_e[:2,2], p_k1_e, linestyle=":", linewidth=0.5)
    # smb.plot_arrow(Tk0_e[:2,2], p_k0_e, linestyle=":", linewidth=0.5)

    # observations
    # ax.set_title("k=1")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # plt.legend()
    plt.tight_layout()
    plt.show()

    