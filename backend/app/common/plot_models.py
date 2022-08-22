import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
plt.style.use('tableau-colorblind10')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def make_plots(df_results, filename):
    # load datafiles
    df_exp_models = pd.read_csv('/nanobody-polyreactivity/app/experiments/low_throughput_polyspecificity_scores_w_exp.csv')
    df_h = pd.read_csv('/nanobody-polyreactivity/app/experiments/high_polyreactivity_high_throughput.csv')
    df_l = pd.read_csv('/nanobody-polyreactivity/app/experiments/low_polyreactivity_high_throughput.csv')
    fig, axes = plt.subplots(2,2,figsize=(12,10))
    axes = axes.ravel()
    for tickm, model in enumerate(['origFACS lr onehot','deepFACS lr onehot']):

        n0 = axes[tickm].hist(df_l[model],bins = 20,alpha = 0.4,label = 'low', density = True)
        n1 = axes[tickm].hist(df_h[model],bins = 20,alpha = 0.4,label = 'high', density = True)
        # if len(df_results) > 100:
            # axes[tickm].hist(df_results[model],bins = 20,alpha = 0.4,label = 'your input', density = True)
        # else:
        axes[tickm].vlines(df_results[model],0,max(n1[0].max(),n0[0].max()), color = 'red', label = 'your input')
        axes[tickm].set_title(f'high-throughput experiment results\n vs. {model} model')
        axes[tickm].legend(title = 'polyreactivity',frameon = False, loc = 'best')
        axes[tickm].set_xlabel('model score')

    x = [0,6.236923,60.022029,100]
    df_exp_models.loc[df_exp_models.loc[:,'Biorep average'] < x[1],'polyreactivity'] = 'minimal'
    df_exp_models.loc[(df_exp_models.loc[:,'Biorep average'] >= x[1]) & (df_exp_models.loc[:,'Biorep average'] < x[2]),'polyreactivity'] = 'moderate'
    df_exp_models.loc[(df_exp_models.loc[:,'Biorep average'] >= x[2]),'polyreactivity'] = 'high'

    # fig,axes = plt.subplots(1,2,figsize=(12,5))
    # axes = axes.ravel()

    color_map = {'high':colors[1],'moderate':colors[0],'minimal': colors[2],'your input': 'r'}
    for tickm,model in enumerate(['origFACS lr onehot','deepFACS lr onehot']):

        df_exp_models.plot.scatter(y = 'Biorep average',x = model,c=[color_map[i] for i in df_exp_models.polyreactivity], ax = axes[tickm+2],s=50)
        axes[tickm+2].vlines(df_results[model],0,df_exp_models['Biorep average'].max(), color = 'red', label = 'your input')

        axes[tickm+2].set_xlabel('model score')
        axes[tickm+2].set_ylabel('polyreactivity (% maximum)')
        legend_items = []
        for key,value in color_map.items():
            legend_items.append(mpatches.Patch(color=value, label=key),)
        axes[tickm+2].legend(handles=legend_items,frameon = False,title = 'Polyreactivity')
        axes[tickm+2].set_title(f'low-throughput experiment results\n vs. {model} model')
    plt.tight_layout()
    plt.savefig(filename, dpi = 300)