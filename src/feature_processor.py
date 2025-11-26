import numpy as np
import pandas as pd

class XGProcessor:
    def __init__(self):
        self.danger_map = {"low": 1, "med": 2, "high": 3}

    # Add geometric features
    def add_distance(self, df):
        df["distance"] = np.sqrt(df["x_coord"]**2 + df["y_coord"]**2)
        return df
    
    def add_shot_angle(self, df):
        """
        Adds a vectorized shot_angle column.
        """

        x = np.abs(df["x_coord"].values)
        y = df["y_coord"].values

        dx = 89 - x

        behind = dx <= 0

        angle = np.degrees(np.arctan2(np.abs(y), np.abs(dx)))
        angle = np.where(behind, 90.0, angle)

        df["shot_angle"] = angle.round(3)
        return df
    
    # Add Shot Value for High, Medium, Low danger scoring chances
    def add_shot_value(self, df):
        df["danger_numeric"] = df["danger_zone"].map(self.danger_map)
        df["shot_value"] = df["danger_numeric"] + df["rebound"] + df["rush"]
        return df
    
    # Add situation flag
    def add_situation(self, df):
        """ 
        Adds a 'situation' column based on home/away skaters.
        """

        df["shooting_team_skaters"] = np.where(df["home"] == 1, df["home_skaters"], df["away_skaters"])
        df["defending_team_skaters"] = np.where(df["home"] == 1, df["away_skaters"], df["home_skaters"])

        df["situation"] = "EV" 

        df.loc[df["shooting_team_skaters"] > df["defending_team_skaters"], "situation"] = "PP"

        df.loc[df["shooting_team_skaters"] < df["defending_team_skaters"], "situation"] = "SH"

        df.drop(columns=["shooting_team_skaters", "defending_team_skaters"], inplace=True)

        return df
    
    def processFenwick(self, df):
        """
        Process raw Fenwick shots into a model ready DataFrame.
        """
        df["shot_on_glove"] = df["shoots"] + df["goalie_catches"]
        df["home_skaters"] = df["home_skaters"].astype(int)
        df["away_skaters"] = df["away_skaters"].astype(int)
        df = df[df["home_skaters"] >= 3]
        df = df[df["away_skaters"] >= 3]
        df = df[df["goalie_id"].notnull()]

        df = self.add_distance(df)
        df = self.add_shot_angle(df)
        df = self.add_shot_value(df)
        df = self.add_situation(df)

        drop_cols = ["game_id", "team_id", "home_def_side", "x_coord", "y_coord", "shooter_id", "shooter", "goalie_id", "goalie", "goalie_catches", "shoots", "zone"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        return df