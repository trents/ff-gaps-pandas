''' Converts raw passing data from football-reference.com
     to fantasy football points. '''

import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    passing_stats = pd.read_csv("passing-2020.csv")
    rushing_stats = pd.read_csv("rushing-2020.csv")
    player_points_list = []
    td_points_list = []
    yard_points_list = []
    int_points_list = []
    tdint_list = []
    shortname_list = []

    # cleaning quirks of data
    passing_stats['Pos'] = passing_stats['Pos'].replace(['qb'],'QB')
    passing_stats = passing_stats.loc[passing_stats["Pos"] == "QB"]

    for index, row in passing_stats.iterrows():
        namesplit = row["Player"].split("\\")
        shortname_list.append(namesplit[1])
    passing_stats["ShortName"] = shortname_list

    shortname_list = []

    for index, row in rushing_stats.iterrows():
        namesplit = row["Player"].split("\\")
        shortname_list.append(namesplit[1])
    rushing_stats["ShortName"] = shortname_list

    passing_stats = pd.merge(left=passing_stats, right=rushing_stats, how='left', left_on='ShortName', right_on='ShortName')

    for index, row in passing_stats.iterrows():
        player_points = 0
        player_points = (row["Yds_x"] / 25 + row["Yds_y"] / 10) / row["G_x"]
        yard_points_list.append((row["Yds_x"] / 25 + row["Yds_y"] / 10) / row["G_x"])
        player_points += (row["TD_x"] * 4 + row["TD_y"] * 6) / row["G_x"]
        td_points_list.append((row["TD_x"] * 4 + row["TD_y"] * 6) / row["G_x"])
        player_points += ((row["Int"] + row["Fmb"]) * -2) / row["G_x"]
        int_points_list.append(((row["Int"] + row["Fmb"]) * -2) / row["G_x"])
        player_points_list.append(player_points)
        tdint_list.append(row["TD_x"] / row["Int"])

    passing_stats["Points"] = player_points_list
    passing_stats["TDPoints"] = td_points_list
    passing_stats["YdPoints"] = yard_points_list
    passing_stats["IntPoints"] = int_points_list
    passing_stats["TDInt"] = tdint_list
    passing_stats.sort_values(by=['Points'],ignore_index=True,inplace=True,ascending=False)

# mormalize columns we care about

    for stat in ['Cmp','Att_x','Yds_x','TD_x','Int','1D_x','Lng_x','QBR','Sk','4QC','GWD','Att_y','Yds_y','TD_y','1D_y','Lng_y','Fmb']:
        max_value = passing_stats[stat].max()
        min_value = passing_stats[stat].min()
        data = []

        for index, row in passing_stats.iterrows():
            if row[stat] == "":
                row[stat] = 0
            data.append((row[stat] - min_value) / row["G_x"])
    
        max_value = max(data)
        min_value = min(data)
    
        for count, _ in enumerate(data):
            data[count] = data[count] / max_value
            if pd.isna(data[count]):
                data[count] = 0

        passing_stats[stat + "Norm"] = data

# cluster the QBs into tiers based on yard, TD, and int fantasy points

    data = passing_stats[['CmpNorm', 'Att_xNorm', 'Yds_xNorm', 'TD_xNorm', 'IntNorm', '1D_xNorm', 'Lng_xNorm', 'QBRNorm', 'SkNorm', '4QCNorm', 'GWDNorm', 'Att_yNorm', 'Yds_yNorm', 'TD_yNorm', '1D_yNorm', 'Lng_yNorm', 'FmbNorm']]

# figure out optimal number of clusters

    sse = []
    for i in range(1,15):
        kmeans = KMeans(
            init="random",
            n_clusters=i,
            n_init=10,
            max_iter=300,
            random_state=1337)

        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        
# find the first index in sse where the gap between it and the next one is less than 1/3 the mean of the gaps

    sse_gaps = []
    for i in range(0,len(sse)-1):
        sse_gaps.append(sse[i] - sse[i+1])
    cutoff = sum(sse_gaps) / (len(sse_gaps) * 2) 

    cutoff_number = 0
    for i in range(0,len(sse_gaps)):
        if cutoff_number == 0 and sse_gaps[i] < cutoff:
            cutoff_number = i + 1

# use the cutoff_number for our analysis

    kmeans = KMeans(
        init="random",
        n_clusters=cutoff_number,
        n_init=10,
        max_iter=300,
        random_state=1337)

    kmeans.fit(data)

    passing_stats["Tiers"] = kmeans.labels_

# make a pretty scatter plot of the tiers, with QBs labeled

    plt.scatter(passing_stats['Points'],passing_stats['TD_xNorm'],c=passing_stats['Tiers'],cmap='rainbow')
    plt.title("2020 NFL QBs, PPG vs Passing TDs")
    plt.xlabel("PPG")
    plt.ylabel("Passing TDs")
    
    for index, row in passing_stats.iterrows():
        label = row["ShortName"]
        plt.annotate(label,(row['Points'],row['TD_xNorm']),textcoords="offset points",xytext=(3,3),ha='left',fontsize=8)
    plt.show()

# clean up tiers

    tiers_means = []

    tiers = passing_stats.Tiers.unique()
    tiers_keyed = {}
    tier_max = len(tiers)

    for item in tiers:
        tiers_keyed[item] = tier_max
        tier_max -= 1

    passing_stats["Tiers"].replace(tiers_keyed, inplace=True)

    passing_stats.sort_values(by=['Tiers','Points'],ignore_index=True,inplace=True,ascending=False)

# experiment with tiers data

    tier_list = passing_stats["Tiers"].unique()
    tier_data = pd.DataFrame(tier_list, columns=['Tiers'])

    for metric in ('CmpNorm', 'Att_xNorm', 'Yds_xNorm', 'TD_xNorm', 'IntNorm', '1D_xNorm', 'Lng_xNorm', 'QBRNorm', 'SkNorm', '4QCNorm', 'GWDNorm', 'Att_yNorm', 'Yds_yNorm', 'TD_yNorm', '1D_yNorm', 'Lng_yNorm', 'FmbNorm'):
        temp_points = passing_stats.groupby('Tiers')[metric].mean()
        tier_data = pd.merge(left=tier_data, right=temp_points, how='left', left_on='Tiers', right_on='Tiers')
        avg = tier_data[metric].mean()
        diff = []
        for index, row in tier_data.iterrows():
            diff.append(row[metric] - avg)

        tier_data[metric + "_Diff"] = diff

    print(tier_data)

# print the tiers

    tier_no = 1
    prev_tier = -1

    for index, row in passing_stats.iterrows():
        if index == 0:
            distinct_stat = ""
            for metric in ('CmpNorm', 'Att_xNorm', 'Yds_xNorm', 'TD_xNorm', 'IntNorm', '1D_xNorm', 'Lng_xNorm', 'QBRNorm', 'SkNorm', '4QCNorm', 'GWDNorm', 'Att_yNorm', 'Yds_yNorm', 'TD_yNorm', '1D_yNorm', 'Lng_yNorm', 'FmbNorm'):
                max_row = tier_data[metric + "_Diff"].max()
                if tier_data.loc[tier_data['Tiers'] == row["Tiers"], metric+"_Diff"].iloc[0] == max_row:
                    distinct_stat = distinct_stat + metric + " "
            print("Group 1 - ",distinct_stat)
            print("-----")
            prev_tier = row["Tiers"]
        elif prev_tier != row["Tiers"]:
            tier_no += 1
            prev_tier = row["Tiers"]
            distinct_stat = ""
            for metric in ('CmpNorm', 'Att_xNorm', 'Yds_xNorm', 'TD_xNorm', 'IntNorm', '1D_xNorm', 'Lng_xNorm', 'QBRNorm', 'SkNorm', '4QCNorm', 'GWDNorm', 'Att_yNorm', 'Yds_yNorm', 'TD_yNorm', '1D_yNorm', 'Lng_yNorm', 'FmbNorm'):
                max_row = tier_data[metric + "_Diff"].max()
                if tier_data.loc[tier_data['Tiers'] == row["Tiers"], metric+"_Diff"].iloc[0] == max_row:
                    distinct_stat = distinct_stat + metric + " "
            print("")
            print("Group",tier_no," - ",distinct_stat)
            print("-----")

        print(index + 1,"- ",row["Player_x"])

if __name__ == "__main__":
    main()
