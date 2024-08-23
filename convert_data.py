""" Prepare data from the SP location model (Raviv 2023) to the simulation
E.g., Convert the lat long locations to UTM and remove type 0 SPs"""

import pandas as pd
from matplotlib import pyplot as plt
from pyproj import Proj


def generate_data(dp_datafile, sp_datafile):
    dp_df: pd.DataFrame = pd.read_csv(dp_datafile)
    sp_df: pd.DataFrame = pd.read_csv(sp_datafile)
    dp_df.columns = dp_df.columns.str.replace(' ', '')
    sp_df.columns = sp_df.columns.str.replace(' ', '')
    sp_df = sp_df[sp_df['type'] != 0]  # remove non-SPs (type 0)
    type_capacity = {1: 30, 2: 60, 3: 100, 4: 150}  # capacity per SP type
    sp_df['capacity'] = sp_df['type'].map(type_capacity).astype(int)

    # convert lat/lon to x/y coordinates
    p = Proj(proj='utm', zone=33, ellps='WGS84', preserve_units=False)
    dp_df['x'], dp_df['y'] = p(dp_df['LON'], dp_df['LAT'])
    sp_df['x'], sp_df['y'] = p(sp_df['LON'], sp_df['LAT'])
    dp_df[['x', 'y']] = dp_df[['x', 'y']].round() / 1000
    sp_df[['x', 'y']] = sp_df[['x', 'y']].round() / 1000

    dp_df.to_csv('output/dp.csv', index=False)
    sp_df.to_csv('output/sp.csv', index=False)
    pivoted_df = dp_df.pivot(index='y', columns='x')
    plt.pcolor(pivoted_df, cmap='coolwarm')
    plt.axis('equal')
    plt.scatter(sp_df['x'], sp_df['y'], color='black', s=5)
    plt.savefig('output/heatmap.png')
    return dp_df, sp_df


if __name__ == '__main__':
    output = generate_data(dp_datafile="data/linz-dp-map_pwl-0.5-0.04-300-10.csv",
                           sp_datafile="data/linz-sp-map_pwl-0.5-0.04-300-10.csv")
    print(output)
