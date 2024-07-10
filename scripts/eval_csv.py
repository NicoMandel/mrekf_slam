import os.path
import numpy as np
from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import seaborn as sns
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.5, "font" : "sans-serif", "font-family" : "Times New Roman"})
filts = ["FP", "EXC", "INC", "MR", "DATMO"]
cols = sns.color_palette("colorblind6", len(filts))
palette = dict(zip(filts, cols))
sns.set_palette("colorblind6")
sns.set_style("ticks")
# sns.despine()


def plot_false_negatives(csvp : str):
    """
        Function to plot the impact of false negatives on the EKF
        FIgure 3
    """

    df2 = prepare_csv(csvp)
    drop_filters(df2, "Filter Type", *["FP", "DATMO", "MR"])
    drop_filters(df2, "metric", *["dyn_ATE", "detP"])

    metrics = ["ate", "SDE", "rotation_dist", "translation_dist"]
    for i, ind in enumerate(metrics):
        plot_df = df2[df2["metric"] == ind]
        ax = sns.lineplot(plot_df, x="static", y="EKF_", hue="Filter Type", palette=palette, errorbar="ci")
        ax.set_xticks(df2["static"].unique()[::3])
        yt = ax.get_yticks()
        nyt = np.arange(0, np.ceil(yt.max()), np.ceil(yt.max()) / 5)
        ax.set_yticks(nyt)
        ax.set_ylabel("")
        sns.despine()
        if i != len(metrics)-1: ax.get_legend().remove()
        plt.tight_layout()
        print(ind)
        plt.show()

def plot_false_positives(csvp : str):
    """
        Function to plot the impact of false positives on the EKF
        Figure 4
    """
    df2 = prepare_csv(csvp)
    drop_filters(df2, "Filter Type", *["INC", "DATMO", "MR"])
    drop_filters(df2, "metric", *["dyn_ATE", "detP", "SDE"])

    metrics = ["ate", "rotation_dist", "translation_dist"]
    for i, ind in enumerate(metrics):
        plot_df = df2[df2["metric"] == ind]
        ax = sns.lineplot(plot_df, x="static", y="EKF_", hue="Filter Type", palette=palette, errorbar="ci")     # tyle="filter_subtype"
        ax.set_xticks(df2["static"].unique()[::3])
        yt = ax.get_yticks()
        nyt = np.arange(0, np.ceil(yt.max()), np.ceil(yt.max()) / 5)
        ax.set_yticks(nyt)
        ax.set_ylabel("")
        sns.despine()
        if i != len(metrics)-1: ax.get_legend().remove()
        plt.tight_layout()
        print(ind)
        plt.show()

def plot_dynamic_ego_ates(csvp: str):
    """
        Figure 5 for FPs
    """

    df2 = prepare_csv(csvp)
    drop_filters(df2, "Filter Type", *["INC", "DATMO", "MR"])
    drop_filters(df2, "metric", *["dyn_ATE", "detP", "SDE", "rotation_dist", "translation_dist"])
    df2["Motion Model"].replace({"None" : "SM"}, inplace=True)

    yts = []
    for ind in df2['dynamic'].unique():
        plot_df = df2[df2["dynamic"] == ind]
        ax = sns.lineplot(plot_df, x="static", y="EKF_", hue="Filter Type", style="Motion Model", palette=palette, errorbar="ci")
        yts.append(ax.get_yticks().max())
        
    nyt = np.arange(0, np.ceil(max(yts)), np.ceil(max(yts)) / 5)    
    print(nyt)
    plt.clf()
    plt.cla()
    for ind in df2['dynamic'].unique():
        # fig = plt.figure(figsize=(15,9))
        plot_df = df2[df2["dynamic"] == ind]
        ax = sns.lineplot(plot_df, x="static", y="EKF_", hue="Filter Type", style="Motion Model", palette=palette, errorbar="ci")
        ax.set_xticks(df2["static"].unique()[::3])
        ax.set_ylabel("")
        ax.set_yticks(nyt)
        sns.despine()
        if ind != df2["dynamic"].unique().max(): ax.get_legend().remove()
        # plt.title(ind)
        plt.tight_layout()
        print(ind)
        plt.show()

def plot_dynamic_cumulative_ates(csvp: str):
    """
        Figure 7
    """
    df2 = prepare_csv(csvp)
    drop_filters(df2, "Filter Type", *["INC", "EXC", "FP"])
    drop_filters(df2, "metric", *["ate", "detP", "SDE", "rotation_dist", "translation_dist"])
    df2["Motion Model"].replace({"None" : "SM"}, inplace=True)

    yts = []
    for ind in df2['dynamic'].unique():
        plot_df = df2[df2["dynamic"] == ind]
        ax = sns.lineplot(plot_df, x="static", y="EKF_", hue="Filter Type", style="Motion Model", palette=palette, errorbar="ci")
        yts.append(ax.get_yticks().max())
        
    nyt = np.arange(0, np.ceil(max(yts)), np.ceil(max(yts)) / 5)    
    print(nyt)
    plt.clf()
    plt.cla()
    for ind in df2['dynamic'].unique():
        # fig = plt.figure(figsize=(15,9))
        plot_df = df2[df2["dynamic"] == ind]
        ax = sns.lineplot(plot_df, x="static", y="EKF_", hue="Filter Type", style="Motion Model", palette=palette, errorbar="ci")
        ax.set_xticks(df2["static"].unique()[::3])
        ax.set_ylabel("")
        ax.set_yticks(nyt)
        sns.despine()
        print(ind)
        if ind != df2["dynamic"].unique().max(): ax.get_legend().remove()
        plt.tight_layout()
        plt.show()

def plot_dynamic_cumulative_sdes(csvp: str):
    """
        Figure 8
    """
    df2 = prepare_csv(csvp)
    drop_filters(df2, "Filter Type", *["INC", "EXC", "FP"])
    drop_filters(df2, "metric", *["ate", "detP", "dyn_ATE", "rotation_dist", "translation_dist"])
    df2["Motion Model"].replace({"None" : "SM"}, inplace=True)

    yts = []
    for ind in df2['dynamic'].unique():
        plot_df = df2[df2["dynamic"] == ind]
        ax = sns.lineplot(plot_df, x="static", y="EKF_", hue="Filter Type", style="Motion Model", palette=palette, errorbar="ci")
        yts.append(ax.get_yticks().max())
        
    nyt = np.arange(0, np.ceil(max(yts)), np.ceil(max(yts)) / 5)    
    print(nyt)
    plt.clf()
    plt.cla()
    for ind in df2['dynamic'].unique():
        # fig = plt.figure(figsize=(15,9))
        plot_df = df2[df2["dynamic"] == ind]
        ax = sns.lineplot(plot_df, x="static", y="EKF_", hue="Filter Type", style="Motion Model", palette=palette, errorbar="ci")
        ax.set_xticks(df2["static"].unique()[::3])
        ax.set_ylabel("")
        ax.set_yticks(nyt)
        sns.despine()
        if ind != df2["dynamic"].unique().max(): ax.get_legend().remove()
        print(ind)
        plt.tight_layout()
        plt.show()

def plot_dynamic_metrics(csvp : str):
    """
        Function to plot the impact of incorporating dynamic metrics into the EKF
        FIgure 6
    """

    df2 = prepare_csv(csvp)
    drop_filters(df2, "Filter Type", *["INC", "FP"])
    drop_filters(df2, "metric", *["detP"])
    drop_filters(df2, "dynamic", *[2, 3, 4, 5])
    df2["Motion Model"].replace({"None" : "SM"}, inplace=True)

    metrics = ["ate", "rotation_dist", "translation_dist"]
    for ind in metrics:
        plot_df = df2[df2["metric"] == ind]
        ax = sns.lineplot(plot_df, x="static", y="EKF_", hue="Filter Type", style="Motion Model", palette=palette, errorbar="ci")
        ax.set_xticks(df2["static"].unique()[::3])
        yt = ax.get_yticks()
        nyt = np.arange(0, np.ceil(yt.max()), np.ceil(yt.max()) / 5)
        ax.set_yticks(nyt)
        ax.set_ylabel("")
        
        sns.despine()
        # ax.set_title(ind)
        print(ind)
        if ind != metrics[-1]: ax.get_legend().remove()
        plt.tight_layout()
        # plt.show()
    
    plt.clf()
    drop_filters(df2, "Filter Type", *["EXC"])
    pldf = df2[df2["metric"] == "dyn_ATE"]
    ax = sns.lineplot(pldf, x="static", y="EKF_", hue="Filter Type", style="Motion Model", palette=palette, errorbar="ci")
    ax.set_xticks(df2["static"].unique()[::3])
    yt = ax.get_yticks()
    nyt = np.arange(0, np.ceil(yt.max()), np.ceil(yt.max()) / 5)
    ax.set_yticks(nyt)
    ax.set_ylabel("")
    ax.get_legend().remove()
    sns.despine()
    print("dyn_ATE")

    # ax.set_title(ind)
    plt.tight_layout()
    plt.show()

def plot_full(csvp : str):
    """
        Function to plot the impact of false positives on the EKF
    """
    df = pd.read_csv(csvp, index_col=0)
    df.drop(['time'], axis=1, inplace=True)
    df['timestamp'] = df.index
    print(df.head())

    sl = ["EKF_EXC", "EKF_FP", "EKF_INC", "EKF_MR"]        # 
    xx = pd.wide_to_long(df, sl, i="timestamp", j="metric", suffix="\D+")
    xx.reset_index(inplace=True)
    stub = ["EKF_"]
    yy = pd.wide_to_long(xx, stub, i=["timestamp", "metric"], j="filter", suffix="\D+")
    zz = yy.reset_index()
    zz.drop("timestamp", axis=1, inplace=True)
    # zz.drop(zz[zz['metric']== "-scale"].index , inplace=True)
    zz.drop(zz[zz['filter']== "FP"].index , inplace=True)
    # ! Uncomment these lines to see differences in different models
    zz.drop(zz[zz['filter']== "INC"].index , inplace=True)
    # zz.to_csv("bigcsv.csv")
    zz.drop(zz[zz['dynamic']== 1].index , inplace=True)
    zz.drop(zz[zz['dynamic']== 4].index , inplace=True)
    zz.drop(zz[zz['dynamic']== 3].index , inplace=True)
    zz.drop(zz[zz['dynamic']== 2].index , inplace=True)
    # zz.drop(zz[zz['motion_model']== "BodyFrame"].index , inplace=True)
    # zz.drop(zz[zz['motion_model']== "KinematicModel"].index , inplace=True)
    g = sns.FacetGrid(zz, col="metric", col_wrap=2, sharey=False)
    g.map_dataframe(sns.lineplot, x="static", y="EKF_", hue="motion_model", style="filter", ci="sd")
    g.add_legend()
    plt.show()

def calculate_means(csvf : str):
    """
        Function to naively calculate the means
    """
    df = pd.read_csv(csvf, index_col=0)
    df.drop(['time', 'fp_count'], axis=1, inplace=True)
    df['timestamp'] = df.index
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)

    sl = ["EKF_EXC", "EKF_INC", "EKF_FP", "EKF_MR"]
    xx = pd.wide_to_long(df, sl, i=["timestamp"], j="metric", suffix="\D+")
    xx.reset_index(inplace=True)
    stub = ["EKF_"]
    yy = pd.wide_to_long(xx, stub, i=["timestamp", "metric"], j="filter", suffix="\D+")
    zz = yy.reset_index()
    zz.rename(columns={'EKF_' : 'value'}, inplace=True)
    zz.drop(['timestamp'], axis=1, inplace=True)
    # ate_mean = df.groupby([])
    cc = zz.set_index(['metric', 'motion_model', 'filter', 'dynamic', 'static', 'seed'])
    #! look here:    https://pandas.pydata.org/docs/user_guide/advanced.html#advanced-indexing-with-hierarchical-index
    # ! and also here: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html 
    # bb = zz.groupby(["metric", "motion_model", "filter", "dynamic", "static"])
    ate_df = cc.loc["-ate"]
    ate_bf_exc = ate_df.loc[("BodyFrame", "EXC")]
    ate_bf_mr = ate_df.loc[("BodyFrame", "MR")]
    ate_km_exc = ate_df.loc[("KinematicModel", "EXC")],
    ate_km_mr = ate_df.loc[("KinematicModel", "MR")]
    ate_sm_exc = ate_df.loc[("StaticModel", "EXC")]
    ate_sm_mr = ate_df.loc[("StaticModel", "MR")]
    # ate_sm_exc.loc[1].median()
    ate_sm_exc.loc[1].std()
    ate_sm_exc.loc[1].mean()
    print("Test Debug line")

def plot_all_models(csvf : str):
    df = pd.read_csv(csvf, index_col=0)
    df.drop(['time'], axis=1, inplace=True)
    df['case'] = df.index
    print(df.head())
    mdls = ["SM", "KM", "BF"]
    sl = ["EKF_EXC", "EKF_INC"]        # 
    sl += ["EKF_FP:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_MR:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_DATMO:{}".format(mdl) for mdl in mdls]
    xx = pd.wide_to_long(df, sl, i="case", j="metric", suffix="\D+")
    xx.reset_index(inplace=True)

    stub = ["EKF_"]
    yy = pd.wide_to_long(xx, stub, i=["case", "metric"], j="filter", suffix="\D+")
    zz = yy.reset_index()
    zz.drop("case", axis=1, inplace=True)
    split_filter = zz['filter'].str.split(':', n=1, expand=True)
    zz['filter_type'] = split_filter[0]
    zz['filter_subtype'] = split_filter[1].fillna('None')

    # Comment for undistorted display
    zz.drop(zz[zz['filter_type']== "FP"].index , inplace=True)
    zz.drop(zz[zz['filter_type']== "DATMO"].index , inplace=True)
    zz.drop(zz[zz['filter_type']== "MR"].index , inplace=True)
    zz.dropna(axis=0, inplace=True)
    # zz.drop(zz[zz['filter_type']== "INC"].index , inplace=True)

    g = sns.FacetGrid(zz, col="metric", col_wrap=2, sharey=False)
    g.map_dataframe(sns.lineplot, x="static", y="EKF_", hue="filter_type", ci="sd")     #style="filter_subtype"
    g.add_legend()
    plt.show()

    print("Test debug line")

def plot_dyn_ates(csvf : str):
    df = pd.read_csv(csvf, index_col=0)
    df.drop(['time'], axis=1, inplace=True)
    df['case'] = df.index
    print(df.head())

    mdls = ["SM", "KM", "BF"]
    sl = ["EKF_EXC", "EKF_INC"]        # 
    sl += ["EKF_FP:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_MR:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_DATMO:{}".format(mdl) for mdl in mdls]
    xx = pd.wide_to_long(df, sl, i="case", j="metric", suffix="\D+")
    xx.reset_index(inplace=True)

    stub = ["EKF_"]
    yy = pd.wide_to_long(xx, stub, i=["case", "metric"], j="filter", suffix="\D+")
    zz = yy.reset_index()
    zz.drop("case", axis=1, inplace=True)
    split_filter = zz['filter'].str.split(':', n=1, expand=True)
    zz['filter_type'] = split_filter[0]
    zz['filter_subtype'] = split_filter[1].fillna('None')

    # Comment for undistorted display
    zz.drop(zz[zz['filter_type']== "FP"].index , inplace=True)
    # zz.drop(zz[zz['filter_type']== "INC"].index , inplace=True)
    zz.drop(zz[zz['filter_type']== "DATMO"].index , inplace=True)
    zz.drop(zz[zz['filter_type']== "MR"].index , inplace=True)

    # dropping all other metrics
    # zz.drop(zz[zz['metric']== "-ate"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-dyn_ATE"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-rotation_dist"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-translation_dist"].index , inplace=True)

    # Plotting
    g = sns.FacetGrid(zz, col="dynamic", col_wrap=3, sharey=True)
    g.map_dataframe(sns.lineplot, x="static", y="EKF_", hue="filter_type", style="filter_subtype", ci="sd")
    g.add_legend()
    plt.show()
    print("Test debug line")

def plot_fp_models(csvf : str):
    df = pd.read_csv(csvf, index_col=0)
    df.drop(['time'], axis=1, inplace=True)
    df['case'] = df.index
    print(df.head())

    mdls = ["SM", "KM", "BF"]
    sl = ["EKF_EXC", "EKF_INC"]        # 
    sl += ["EKF_FP:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_MR:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_DATMO:{}".format(mdl) for mdl in mdls]
    xx = pd.wide_to_long(df, sl, i="case", j="metric", suffix="\D+")
    xx.reset_index(inplace=True)

    stub = ["EKF_"]
    yy = pd.wide_to_long(xx, stub, i=["case", "metric"], j="filter", suffix="\D+")
    zz = yy.reset_index()
    zz.drop("case", axis=1, inplace=True)
    split_filter = zz['filter'].str.split(':', n=1, expand=True)
    zz['filter_type'] = split_filter[0]
    zz['filter_subtype'] = split_filter[1].fillna('None')

    # Comment for undistorted display
    # zz.drop(zz[zz['filter_type']== "FP"].index , inplace=True)
    zz.drop(zz[zz['filter_type']== "INC"].index , inplace=True)
    zz.drop(zz[zz['filter_type']== "DATMO"].index , inplace=True)
    zz.drop(zz[zz['filter_type']== "MR"].index , inplace=True)
    # zz.drop(zz[zz['filter_type']== "EXC"].index , inplace=True)

    # dropping all other metrics
    zz.drop(zz[zz['metric']== "-dyn_ate"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-rotation_dist"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-translation_dist"].index , inplace=True)

    # Plotting
    g = sns.FacetGrid(zz, col="dynamic", col_wrap=3, sharey=True)
    g.map_dataframe(sns.lineplot, x="static", y="EKF_", hue="filter_type", style="filter_subtype", ci="sd")
    g.add_legend()
    plt.show()
    print("Test debug line")

def plot_sdes(csvf : str):
    df = pd.read_csv(csvf, index_col=0)
    df.drop(['time'], axis=1, inplace=True)
    df['case'] = df.index
    print(df.head())

    mdls = ["SM", "KM", "BF"]
    sl = ["EKF_EXC", "EKF_INC"]        # 
    sl += ["EKF_FP:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_MR:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_DATMO:{}".format(mdl) for mdl in mdls]
    xx = pd.wide_to_long(df, sl, i="case", j="metric", suffix="\D+")
    xx.reset_index(inplace=True)

    stub = ["EKF_"]
    yy = pd.wide_to_long(xx, stub, i=["case", "metric"], j="filter", suffix="\D+")
    zz = yy.reset_index()
    zz.drop("case", axis=1, inplace=True)
    split_filter = zz['filter'].str.split(':', n=1, expand=True)
    zz['filter_type'] = split_filter[0]
    zz['filter_subtype'] = split_filter[1].fillna('None')

    # Comment for undistorted display
    zz.drop(zz[zz['filter_type']== "FP"].index , inplace=True)
    # zz.drop(zz[zz['filter_type']== "INC"].index , inplace=True)
    zz.drop(zz[zz['filter_type']== "EXC"].index , inplace=True)

    # dropping all other metrics
    zz.drop(zz[zz['metric']== "-ate"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-rotation_dist"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-translation_dist"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-dyn_ATE"].index , inplace=True)

    # Plotting
    g = sns.FacetGrid(zz, col="dynamic", col_wrap=3, sharey=True)
    g.map_dataframe(sns.lineplot, x="static", y="EKF_", hue="filter_type", style="filter_subtype", ci="sd")
    g.add_legend()
    plt.tight_layout()
    plt.savefig('sdeinc.png', format="png", dpi=300)
    # plt.show()
    print("Test debug line")

def find_interesting_cases(csvf : str):
    df = pd.read_csv(csvf, index_col=0)
    df.drop(['time'], axis=1, inplace=True)
    df['timestamp'] = df.index
    print(df.head())
    mdls = ["SM", "KM", "BF"]
    sl = ["EKF_EXC", "EKF_INC"]        # 
    sl += ["EKF_FP:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_MR:{}".format(mdl) for mdl in mdls]
    xx = pd.wide_to_long(df, sl, i="timestamp", j="metric", suffix="\D+")
    xx.reset_index(inplace=True)

    xx_ate = xx[xx.metric == "-ate"]
    min_col = "EKF_FP:BF"
    aa = xx_ate[xx_ate[min_col] == xx_ate[min_col].min()]
    print(aa)

    diffcol_1 = "EKF_EXC"
    diffcol_2 = "EKF_MR:BF"
    xx_ate["diff"] = xx_ate[diffcol_1] - xx_ate[diffcol_2]
    xx_ate[xx_ate["diff"] == xx_ate["diff"].max()]
    xx_ate.sort_values(by=["diff"], ascending=False)

    # 2 static are the weirdest ones. Have the whole range!
    xx_ate_2 = xx_ate[xx_ate.static == 2]

    stub = ["EKF_"]
    yy = pd.wide_to_long(xx, stub, i=["timestamp", "metric"], j="filter", suffix="\D+")
    zz = yy.reset_index()
    zz.drop("timestamp", axis=1, inplace=True)
    split_filter = zz['filter'].str.split(':', n=1, expand=True)
    zz['filter_type'] = split_filter[0]
    zz['filter_subtype'] = split_filter[1].fillna('None')

def table_summary_case(csvf : str, static_no : int, outpath : str = None):
    df = pd.read_csv(csvf, index_col=0)
    df.drop(['time'], axis=1, inplace=True)
    df['case'] = df.index
    print(df.head())

    mdls = ["SM", "KM", "BF"]
    sl = ["EKF_EXC", "EKF_INC"]        # 
    sl += ["EKF_FP:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_MR:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_DATMO:{}".format(mdl) for mdl in mdls]
    xx = pd.wide_to_long(df, sl, i="case", j="metric", suffix="\D+")
    xx.reset_index(inplace=True)

    stub = ["EKF_"]
    yy = pd.wide_to_long(xx, stub, i=["case", "metric"], j="filter", suffix="\D+")
    zz = yy.reset_index()
    zz.drop("case", axis=1, inplace=True)
    zz.drop("fp_count", axis=1, inplace=True)
    split_filter = zz['filter'].str.split(':', n=1, expand=True)
    zz['filter_type'] = split_filter[0]
    zz['filter_subtype'] = split_filter[1].fillna('None')

    # Drop irrelevant filters
    zz.drop(zz[zz['filter_type']== "FP"].index , inplace=True)
    zz.drop(zz[zz['filter_type']== "INC"].index , inplace=True) 
    zz.drop(zz[zz['filter_type']== "EXC"].index , inplace=True) 

    # dropping all other metrics
    zz.drop(zz[zz['metric']== "-ate"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-rotation_dist"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-translation_dist"].index , inplace=True)

    ########## new section for summaries
    # aa = zz.pivot(index=["static", "dynamic", "timestamp"], columns=["filter", "metric"], values="EKF_") # Alternative
    # zz.pivot_table(index=["static", "dynamic"], columns=["filter", "metric"], values="EKF_", aggfunc=[np.mean, np.std])
    zz["Normalized Dyn-ATE"] = zz["EKF_"] / zz["dynamic"]
    zz.drop("metric", axis=1, inplace=True)
    zz.drop("filter", axis=1, inplace=True)
    zz.drop("EKF_", axis=1, inplace=True)

    # Done with preparation. two versions to get the same data now:
    bb = zz.pivot_table(index=["static", "filter_subtype", "filter_type"], columns=["dynamic"], values="Normalized Dyn-ATE", aggfunc=[np.mean, np.std])

    mst = ["mean" , "std"]
    cc = zz.groupby(["static", "filter_subtype", "filter_type", "dynamic"]).agg({"Normalized Dyn-ATE" : mst})
    bbs = bb.loc[static_no]
    ccs = cc.loc[static_no]
    res = bbs.swaplevel(0,1, axis=1).sort_index(axis=1)
    print("Mean and std.dev from Normalized dynamic EKF")
    print(res)
    if outpath is not None:
        print("saving to: {}".format(outpath))
        res.to_csv(outpath, float_format="%.5f")
        # with pd.ExcelWriter(outpath) as writer:
        #     res.to_excel(writer, sheet_name="results", float_format="%.5f")
        #     wb = writer.book
        #     ws = writer.sheets['results']
        #     ws.number_format

        #     format1 = wb.add_format({'num_format': '0.00000'})
        #     ws.set_row(1, 10, format1)
    print("Test Debug line")

def plot_Pnorm(csvf : str):
    df = pd.read_csv(csvf, index_col=0)
    df.drop(['time'], axis=1, inplace=True)
    df['case'] = df.index
    print(df.head())

    mdls = ["SM", "KM", "BF"]
    sl = ["EKF_EXC", "EKF_INC"]        # 
    sl += ["EKF_FP:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_MR:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_DATMO:{}".format(mdl) for mdl in mdls]
    xx = pd.wide_to_long(df, sl, i="case", j="metric", suffix="\D+")
    xx.reset_index(inplace=True)

    stub = ["EKF_"]
    yy = pd.wide_to_long(xx, stub, i=["case", "metric"], j="filter", suffix="\D+")
    zz = yy.reset_index()
    zz.drop("case", axis=1, inplace=True)
    split_filter = zz['filter'].str.split(':', n=1, expand=True)
    zz['filter_type'] = split_filter[0]
    zz['filter_subtype'] = split_filter[1].fillna('None')

    # Comment for undistorted display
    zz.drop(zz[zz['filter_type']== "FP"].index , inplace=True)
    # zz.drop(zz[zz['filter_type']== "INC"].index , inplace=True)
    # zz.drop(zz[zz['filter_type']== "DATMO"].index , inplace=True)
    # zz.drop(zz[zz['filter_type']== "MR"].index , inplace=True)

    # dropping all other metrics
    # zz.drop(zz[zz['metric']== "-ate"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-dyn_ATE"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-rotation_dist"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-translation_dist"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-SDE"].index , inplace=True)
    zz.drop(zz[zz['metric']== "-ate"].index , inplace=True)


    zz.drop(zz[zz["static"] <= 18].index, inplace=True)
    # Plotting
    g = sns.FacetGrid(zz, col="dynamic", col_wrap=3, sharey=True)
    g.map_dataframe(sns.lineplot, x="static", y="EKF_", hue="filter_type", style="filter_subtype", ci="sd")
    g.add_legend()
    plt.show()
    print("Test debug line")

def prepare_csv(csvf : str) -> pd.DataFrame:
    df = pd.read_excel(open(csvf, 'rb'), header=0)
    df.drop(['time'], axis=1, inplace=True)
    df['case'] = df.index

    mdls = ["SM", "KM", "BF"]
    sl = ["EKF_EXC", "EKF_INC"]        # 
    sl += ["EKF_FP:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_MR:{}".format(mdl) for mdl in mdls]
    sl += ["EKF_DATMO:{}".format(mdl) for mdl in mdls]
    xx = pd.wide_to_long(df, sl, i="case", j="metric", suffix="\D+")
    xx.reset_index(inplace=True)

    stub = ["EKF_"]
    yy = pd.wide_to_long(xx, stub, i=["case", "metric"], j="filter", suffix="\D+")
    zz = yy.reset_index()
    zz.drop("case", axis=1, inplace=True)
    split_filter = zz['filter'].str.split(':', n=1, expand=True)
    zz['filter_type'] = split_filter[0]
    zz['filter_subtype'] = split_filter[1].fillna('None')
    zz["metric"] = zz["metric"].map(lambda x : x.lstrip("-"))
    zz.rename(columns={"filter_type": "Filter Type", "filter_subtype": "Motion Model"}, inplace=True)
    zz.drop(['filter'], axis=1, inplace=True)
    return zz

def drop_filters(df : pd.DataFrame, ind : str, *filters) -> None:
    """
        Drops specific filters or metrics inplace.
        use with "metric" and ["-dyn_ATE", "-rotation_dist", ...] 
        or:
        'filter_type' and  ["FP", "EXC"]
    """
    for fil in filters:
        df.drop(df[df[ind]== fil].index , inplace=True)


if __name__=="__main__":

    # file setup
    fdir = os.path.dirname(__file__)
    basedir = os.path.abspath(os.path.join(fdir, '..'))
    resultsdir = os.path.join(basedir, 'results')
    tmpdir = os.path.join(basedir, '.tmp')
    rescsv = os.path.join(resultsdir, 'ate_2to20.csv')

    fn_csv = os.path.join(resultsdir, 'false_negative.csv')
    fullxlsx = os.path.join(resultsdir, "fullxlsx_20240709.xlsx")
    # plot_false_negatives(fullxlsx)
    # plot_false_positives(fullxlsx)
    # plot_dynamic_ego_ates(fullxlsx)
    # plot_dynamic_metrics(fullxlsx)
    # plot_dynamic_cumulative_ates(fullxlsx)
    plot_dynamic_cumulative_sdes(fullxlsx)

    # fp_csv = os.path.join(resultsdir, 'false_positive.csv')
    # plot_false_positives(fp_csv)

    full_csv = os.path.join(resultsdir, 'full_eval.csv')
    # plot_full(full_csv)
    short_csv = os.path.join(resultsdir, 'short_eval_ate.csv')
    # plot_full(short_csv)
    # calculate_means(full_csv)
    # calculate_means(short_csv)

    all_csv = os.path.join(resultsdir, "all_models_20.csv")
    # find_interesting_cases(all_csv)
    # plot_all_models(all_csv)

    # true initionalisation csv
    tmpdir = os.path.join(basedir, '.tmp')
    true_init_csv = os.path.join(tmpdir, 'debug_2_true_vals', 'debug_2_true_vals.csv')
    fp_csv = os.path.join(tmpdir, "fptest_20240604.csv")
    # plot_fp_models(fp_csv)
    pnorm_csv = os.path.join(tmpdir, 'pnormtest_20240702.csv')
    # plot_Pnorm(pnorm_csv)

    hydratest_csv = os.path.join(tmpdir, 'hydratest_20240517.csv')
    table_summary_case(hydratest_csv, 15, os.path.join(basedir, "15.csv"))
    # plot_all_models(hydratest_csv)
    # plot_dyn_ates(hydratest_csv)

    plot_sdes(sdetest_csv)

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
