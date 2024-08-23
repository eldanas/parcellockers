"""
Simulation output analysis

This script reads the output of pickups.py from the output folder, generates statistics and visualizations.
It also performs a t-test to compare the performance of the different policies.
"""

import os.path
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import product, combinations
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import matplotlib.patches as mpatches

policies = {
    'bl1': 'BL1',
    # 'bl2': 'BL2',
    'bl3': 'BL3',
    'preference_pessimistic': 'PB-PESSIMISTIC',
    'preference_expected': 'PB-EXPECTED',
}

# Parameter
arrival_rates = [0.04, 0.048]
significance_level = 0.05
metrics = ['disutility', 'waiting_time', 'distance']
statistics = ['mean', 'std', 'min', 'max', '50%']
sp_std_values = [0.25, 0.125]
delay_std_values = [0.25, 0.125]


WARMUP_DAYS = 10

os.makedirs("output/analysis", exist_ok=True)

# Generate statistics
for arrival_rate in arrival_rates:
    max_mean_metrics = {met: 0 for met in metrics}
    stats = []
    stats_grouped = []
    for sp_std, delay_std in product(sp_std_values, delay_std_values):
        print("\033[92m")
        print(f"sp_std={sp_std}, delay_std={delay_std}")
        print("\033[0m")
        prefix = f"output/linz_{arrival_rate}_5-2.0_{sp_std}_{delay_std}_"
        parcel_dfs = []
        lower_bound = None
        filename = f"{prefix}lower_bound.csv"
        if not os.path.exists(filename):
            print(f"No lower bound found for {sp_std}, {delay_std}")
            continue
        lower_bound_df = pd.read_csv(filename)
        disutility_lower_bound = lower_bound_df['disutility'].mean()

        for policy in policies.keys():
            filename = f"{prefix}{policy}.csv"
            if os.path.exists(filename):
                print(f"Processing {filename}")
                parcel_df = pd.read_csv(filename, dtype={'arrival_day': int})
                parcel_df = parcel_df[parcel_df['arrival_day'] >= WARMUP_DAYS]
                # Subtract disutility lower bound
                parcel_df['disutility'] -= disutility_lower_bound
                mean_disutility = parcel_df['disutility'].mean()
                for met in metrics:
                    max_mean_metrics[met] = max(max_mean_metrics[met], parcel_df[met].mean())
                parcel_df['policy'] = policies[policy]
                parcel_df = parcel_df.sort_values(by='id')
                parcel_dfs.append(parcel_df)
                stats.append({
                    'sp_std': sp_std,
                    'delay_std': delay_std,
                    'policy': policy,
                    'num_parcels': len(parcel_df),
                    **{f'{met}_{stat}': parcel_df[met].describe()[stat] for met in metrics for stat in statistics},
                    **{f'{met}_confidence_interval': parcel_df[met].sem() * 1.96 for met in metrics},
                })

                parcel_df['arrival_day_bin'] = parcel_df['arrival_day'].apply(lambda x: int(x / 10))
                parcel_df_grouped = parcel_df.groupby('arrival_day_bin')
                for group_name, group in parcel_df_grouped:
                    stats_grouped.append({
                        'sp_std': sp_std,
                        'delay_std': delay_std,
                        'policy': policy,
                        'arrival_day_start': group['arrival_day'].min(),
                        'arrival_day_end': group['arrival_day'].max(),
                        'num_parcels': len(group),
                        **{f'{met}_{stat}': group[met].describe()[stat] for met in metrics for stat in statistics},
                    })
            else:
                print(f"Skipping {filename}")

        # Duncan's test
        combined_parcel_df = pd.concat(parcel_dfs, ignore_index=True)
        pandas2ri.activate()  # Activate the automatic conversion between pandas dataframes and R data.frames
        agricolae = importr('agricolae')  # Import R packages
        r_df = pandas2ri.py2rpy(combined_parcel_df)  # Transfer the DataFrame to R
        robjects.r.assign("dataframe", r_df)
        # Perform ANOVA
        anova_results = robjects.r('''
          model <- aov(disutility ~ policy, data=dataframe)
          model
        ''')
        # Perform Duncan's Test
        duncan_results = robjects.r('''
          results <- duncan.test(model, "policy")
          results
        ''')
        output_file = f"output/analysis/duncan_{arrival_rate}_{sp_std}_{delay_std}.txt"
        with open(output_file, "w") as f:
            f.write(str(duncan_results))

    df = pd.DataFrame(stats).sort_values(by=['sp_std', 'delay_std', 'policy'])
    df.to_csv(f"output/analysis/{arrival_rate}_results.csv", index=False)
    df_grouped = pd.DataFrame(stats_grouped).sort_values(by=['sp_std','delay_std', 'policy', 'arrival_day_start'])
    df_grouped.to_csv(f"output/analysis/{arrival_rate}_results_grouped.csv", index=False)

    # Create plots
    plt.style.use('bmh')
    palette = sns.color_palette("muted", len(policies.keys()))
    df = df.sort_values(by=['policy', 'sp_std', 'delay_std'])
    for (sp_std, delay_std), group in df.groupby(['sp_std', 'delay_std']):
        bar_positions = np.arange(len(group)) * 0.8
        for metric in metrics:
            # remove grid and background
            sns.set(style="whitegrid")
            plt.grid(False)
            plt.rc('font', size=14)
            plt.rc('axes', titlesize=16)
            # Bar colors are based on the alpha value (distance weight)
            # colors = ['red' if alpha == distance_weights[0] else 'blue' for alpha in group['distance_weight']]
            plt.bar(bar_positions, group[f'{metric}_mean'],
                    width=0.4,
                    edgecolor='black',
                    )
            plt.xticks(bar_positions, [policies[p] for p in group['policy']], rotation=45, ha='center')
            plt.gcf().autofmt_xdate()

            max_y = max_mean_metrics[metric]
            decimal_places = 2 if max_y > 0.1 else 4
            max_y *= 1.25

            for i, v in enumerate(group[f'{metric}_mean']):
                text_y = v + (0.01 if max_y > 0.1 else 0.0001)
                plt.text(bar_positions[i], text_y, f"{v:.{decimal_places}f}", ha='center')

            plt.xlabel('Policy')
            units = 'days' if metric == 'waiting_time' else 'km' if metric == 'distance' else ''
            if metric == 'disutility':
                plt.ylabel(f"Mean excess disutility")
            elif metric == 'waiting_time':
                plt.ylabel(f"Mean delay (days)")
            else:
                plt.ylabel(f"Mean distance (km)")

            plt.ylim(0, max_y)  # Set y-axis limit based on the max value

            plt.savefig(f"output/analysis/{arrival_rate}_{sp_std}_{delay_std}_{metric}.png", bbox_inches='tight')
            plt.close()

    # Perform t-tests
    with open(f"output/analysis/{arrival_rate}_results_comparison.csv", "w") as f:
        f.write("policy1,policy2,sp_std,delay_std,t,p,better_policy\n")
        for policy1, policy2 in combinations(policies.keys(), 2):
            print(f"Comparing {policy1} and {policy2}")
            for sp_std, delay_std in product(sp_std_values, delay_std_values):
                prefix = f"output/linz_{arrival_rate}_5-2.0_{sp_std}_{delay_std}"
                file1, file2 = f"{prefix}_{policy1}.csv", f"{prefix}_{policy2}.csv"
                if os.path.exists(file1) and os.path.exists(file2):
                    df1, df2 = pd.read_csv(file1), pd.read_csv(file2)
                    df1, df2 = df1.sort_values(by='id'), df2.sort_values(by='id')
                    t, p = ttest_rel(df1['disutility'], df2['disutility'])
                    result = 0 if p > significance_level else (1 if t < 0 else 2)
                    f.write(f"{policy1},{policy2},{sp_std},{delay_std},{t},{p},{result}\n")
