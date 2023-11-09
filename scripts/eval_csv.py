import os.path 
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


if __name__=="__main__":

    # file setup
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    resultsdir = os.path.join(basedir, 'results')
    rescsv = os.path.join(resultsdir, 'ate.csv')

    # read in csv file
    df = pd.read_csv(rescsv, index_col=0)
    df['timestamp'] = df.index
    df.reset_index(inplace=True, drop=True)
    df['stat_dyn_ratio'] = df['static'] / df['dynamic']
    # df.groupby(['static', 'dynamic',"timestamp"])
    # df.set_index(['dynamic', 'static', 'timestamp'], inplace=True)
    print(df)
    # dfm = df.groupby(["static", "dynamic"])["EKF_EXC", "EKF_INC", "EKF_FP", "EKF_MR"].mean()
    # dfs = df.groupby(["static", "dynamic"])["EKF_EXC", "EKF_INC", "EKF_FP", "EKF_MR"].std()
    mst = ["mean"] #, "std"]
    dfa = df.groupby(["static", "dynamic"]).agg({"EKF_EXC" : mst, "EKF_INC" : mst, "EKF_FP": mst, "EKF_MR" : mst})
    # dfaf = os.path.join(resultsdir, "multiindex.xlsx")
    # writer = pd.ExcelWriter(dfaf)
    # dfa.to_excel(writer, 'ate')
    # writer.save()
    # writer.close()
    dfb = df.groupby(['stat_dyn_ratio']).agg({"EKF_EXC" : mst, "EKF_INC" : mst, "EKF_FP": mst, "EKF_MR" : mst})
    # print(dfa)
    # print(dfb)
    dfa : pd.DataFrame
    dfa.fillna(0., inplace=True)
    dfb.fillna(0., inplace=True)
    for cn in dfb.columns.get_level_values(0):
        dfb[cn, 'div'] = dfb[cn]["mean"] / dfb["EKF_EXC"]["mean"]
        dfa[cn, 'div'] = dfa[cn]["mean"] / dfa["EKF_EXC"]["mean"]
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
    print(dfc)
    sns.heatmap(dfc["div"], norm=LogNorm())
    # sns.clustermap(dfa, norm=LogNorm())
    plt.show()
    # plot some results
