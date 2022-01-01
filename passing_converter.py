''' Converts raw passing data from football-reference.com
     to fantasy football points. '''

import pandas as pd

def main():
    passing_stats = pd.read_csv("passing-2020.csv")
    player_points_list = []

    for index, row in passing_stats.iterrows():
        player_points = 0
        player_points = row["Yds"] / 25
        player_points += row["TD"] * 4
        player_points += row["Int"] * -2
        player_points_list.append(player_points)

    passing_stats["Points"] = player_points_list
    passing_stats.sort_values(by=['Points'],ignore_index=True,inplace=True,ascending=False)

    

    print(passing_stats)

if __name__ == "__main__":
    main()
