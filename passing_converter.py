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
    player_points_list = []

    # cleaning quirks of data
    passing_stats['Pos'] = passing_stats['Pos'].replace(['qb'],'QB')
    passing_stats = passing_stats.loc[passing_stats["Pos"] == "QB"]

    for index, row in passing_stats.iterrows():
        player_points = 0
        player_points = row["Yds"] / 25
        player_points += row["TD"] * 4
        player_points += row["Int"] * -2
        player_points_list.append(player_points)

    passing_stats["Points"] = player_points_list
    passing_stats.sort_values(by=['Points'],ignore_index=True,inplace=True,ascending=False)
    
#    plt.scatter(passing_stats['Yds'],passing_stats['TD'])
#    plt.title("2020 NFL QBs, Yards vs TDs")
#    plt.xlabel("Yards")
#    plt.ylabel("TDs")
#    plt.show()

# cluster the QBs into tiers based on yards and passing TDs

    data = passing_stats.iloc[:,11:13]

    kmeans = KMeans(
        init="random",
        n_clusters=5,
        n_init=10,
        max_iter=300,
        random_state=1337)

    kmeans.fit(data)

    passing_stats["Tiers"] = kmeans.labels_

# make a pretty scatter plot of the tiers, with QBs labeled

    plt.scatter(passing_stats['Yds'],passing_stats['TD'],c=passing_stats['Tiers'],cmap='rainbow')
    plt.title("2020 NFL QBs, Yards vs TDs")
    plt.xlabel("Yards")
    plt.ylabel("TDs")
    
    for index, row in passing_stats.iterrows():
        label = row["Player"]
        clean = label.split(" ")
        clean2 = clean[1].split("\\")
        label = clean2[0]
        plt.annotate(label,(row['Yds'],row['TD']),textcoords="offset points",xytext=(3,3),ha='left',fontsize=8)
    plt.show()

if __name__ == "__main__":
    main()
