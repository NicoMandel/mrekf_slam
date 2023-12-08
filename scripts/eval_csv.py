import os.path 
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__=="__main__":

    # file setup
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    resultsdir = os.path.join(basedir, 'results')
    rescsv = os.path.join(resultsdir, 'ate_2to20.csv')

    # read in csv file
    df = pd.read_csv(rescsv, index_col=0)
    # df.set_index(["timestamp"], inplace=True)
    
    ###### NEW NEW NEW
    df['timestamp'] = df.index
    sl = ["EKF_EXC", "EKF_INC", "EKF_FP", "EKF_MR"]
    xx = pd.wide_to_long(df, sl, i="timestamp", j="metric", suffix="\D+")
    xx.reset_index(inplace=True)
    stub = ["EKF_"]
    yy = pd.wide_to_long(xx, stub, i=["timestamp", "metric"], j="filter", suffix="\D+")
    zz = yy.reset_index()
    # abc =  "-translation_dist"
    abc = "-ate"
    # abc = "-scale"
    # abc = "-rotation_dist"
    sns.barplot(zz.query("metric==@abc"), x="static", y="EKF_", hue="filter", errorbar="sd")
    plt.show()
    # aa = zz.pivot(index=["static", "dynamic", "timestamp"], columns=["filter", "metric"], values="EKF_") # Alternative
    mst = ["mean" , "std"]
    bb = zz.groupby(["dynamic", "static", "metric", "filter"]).agg({"EKF_" : mst})
    cc = bb.reset_index()
    dd = cc.columns.map(''.join)
    cc.columns = dd
    ee =cc.pivot(index=["dynamic", "static"], columns=["filter", "metric"], values=["EKF_mean", "EKF_std"])
    ff = ee.swaplevel(0,2, axis=1)
    gg = ff.sort_index(axis=1)


    ###### OLD
    # df.reset_index(inplace=True, drop=True)
    df['stat_dyn_ratio'] = df['static'] / df['dynamic']

    #### Plotting -> selecting the right metric:
    dd = gg['-ate']
    ff = dd.xs('EKF_mean', axis=1, level=1)
    hh = dd.xs("EKF_std", axis=1, level=1)
    plt.bar(ff)
    sns.barplot(ff, x="EKF_mean", errorbar="EKF_std", norm=Normalize(vmax=0.4))
    plt.show()

    # df.groupby(['static', 'dynamic',"timestamp"])
    # df.set_index(['dynamic', 'static', 'timestamp'], inplace=True)
    print(df)
    # dfm = df.groupby(["static", "dynamic"])["EKF_EXC", "EKF_INC", "EKF_FP", "EKF_MR"].mean()
    # dfs = df.groupby(["static", "dynamic"])["EKF_EXC", "EKF_INC", "EKF_FP", "EKF_MR"].std()
    mst = ["mean" , "std"]
    dfa = df.groupby(["dynamic", "static"]).agg({"EKF_EXC" : mst,  "EKF_FP": mst, "EKF_MR" : mst})      # "EKF_INC" : mst,
    print(dfa)
    
    # g = sns.JointGrid(data=df, x="static", y="dynamic", space=0)
    # g.plot_joint(sns.kdeplot, fill=True,thresh=0)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(df["static"], df["dynamic"], dfa.unstack(level=0)["EKF_EXC"]["mean"], 
                        rstride=1, cstride=1, linewidth=0, antialiased=False
                        )
    plt.show()
    # dfaf = os.path.join(resultsdir, "multiindex.xlsx")
    # writer = pd.ExcelWriter(dfaf)
    # dfa.to_excel(writer, 'ate')
    # writer.save()
    # writer.close()
    dfb = df.groupby(['stat_dyn_ratio']).agg({"EKF_EXC" : mst,  "EKF_FP": mst, "EKF_MR" : mst})     #"EKF_INC" : mst,
    # print(dfa)
    # print(dfb)
    dfa : pd.DataFrame
    dfa.fillna(0., inplace=True)
    dfb.fillna(0., inplace=True)
    # for cn in dfb.columns.get_level_values(0):
        # dfb[cn, 'div'] = dfb[cn]["mean"] / dfb["EKF_EXC"]["mean"]
        # dfa[cn, 'div'] = dfa[cn]["mean"] / dfa["EKF_EXC"]["mean"]
    dfb.sort_index(axis=1, inplace=True)
    dfa.sort_index(axis=1, inplace=True)
    # print(dfb)
    # do some multiindexing
    # sns.barplot(df)
    # sns.catplot(data=df,
    #             kind='bar',
    #             x='static',
    #             y="dynamic",
    #             )
    # df2 = df.pivot(index="static", columns="dynamic", values="EKF_EXC")
    dfc = dfa.swaplevel(0,1, axis=1).sort_index(axis=1)
    # dfc = dfa
    print(dfc)
    print(dfb)
    sns.heatmap(dfc, norm=LogNorm())
    # sns.clustermap(dfa, norm=LogNorm())
    plt.show()
    # plot some results
