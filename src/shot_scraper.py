from itertools import product
import pandas as pd
import logging
from tqdm import tqdm
from nhlpy import NHLClient

# We will need to scrape play-by-play data for the desired information, since we are doing this we might as well build new xG models to try and add more features. Create a class for the scraper.
class NHLShotScraper:

    def __init__(self, seasons=[20232024, 20242025, 20252026]):
        self.client = NHLClient()
        self.seasons = seasons
        self.game_ids = set()
        self.player_dict = {}
        self.player_stats_cache = {}

        self.logger = logging.getLogger("NHLShotScraper")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.logger.info("NHLShotScraper initialized")
        
    def second_diff(self, time1, time2):
        """Calculates the time difference between the occurrence of two events."""
        m1, s1 = int(time1[:2]), int(time1[3:5])
        m2, s2 = int(time2[:2]), int(time2[3:5])
        return abs((m2*60 + s2) - (m1*60 + s1))
    
    def get_game_ids(self):
        """Fetches game_ids with caching."""
        teams = self.client.teams.teams()
        team_abbrs = []

        team_abbrs = [team["abbr"] for team in teams]

        for abbrev, season in product(team_abbrs, self.seasons):
            if abbrev == "UTA" and season < 20242025:
                abbrev = "ARI"

            games = self.client.schedule.team_season_schedule(team_abbr=abbrev, season=season)["games"]
            self.game_ids.update(game["id"] for game in games if game["gameType"] != 1)

        return list(self.game_ids)
    
    def get_player_stats(self, player_id):
        """Fetch player stats with caching."""
        if player_id in self.player_dict:
            return self.player_dict[player_id], self.player_stats_cache[player_id]

        player_info = self.client.stats.player_career_stats(player_id)

        name = player_info["firstName"]["default"] + " " + player_info["lastName"]["default"]

        if player_info["position"] == "G":
            stats = [
                player_info["position"],
                player_info["shootsCatches"],
                player_info["featuredStats"]["regularSeason"]["career"].get("savePctg", None)
            ]
        else:
            stats = [
                player_info["position"],
                player_info["shootsCatches"],
                player_info["featuredStats"]["regularSeason"]["career"].get("shootingPctg", None)
            ]

        self.player_dict[player_id] = name
        self.player_stats_cache[player_id] = stats

        return name, stats

    def is_rebound(self, play, prev_play):
        """Determines if a shot is a rebound or not."""
        if not prev_play:
            return 0
        time_diff = self.second_diff(play["timeInPeriod"], prev_play["timeInPeriod"])
        if prev_play["typeDescKey"] == "blocked-shot" and time_diff <= 2:
            return 1
        if prev_play["typeDescKey"] in ["missed-shot", "shot-on-goal"] and time_diff <= 3:
            return 1
        return 0
    
    def is_rush(self, play, prev_play):
        """Determines if a shot is a rush chance or not"""
        if not prev_play:
            return 0
        
        stoppages = {"stoppage", "faceoff", "goal", "penalty", "period-start", "period-end", "game-end"}
        
        if prev_play["typeDescKey"] in stoppages:
            return 0
        
        time_diff = self.second_diff(play["timeInPeriod"], prev_play["timeInPeriod"])
        if time_diff > 4:
            return 0
        
        shooting_team_id = play["details"].get("eventOwnerTeamId")

        prev_details = prev_play.get("details", {})
        prev_zone = prev_details.get("zoneCode")
        prev_owner = prev_details.get("eventOwnerTeamId")

        if prev_zone is None:
            return 0
        
        if prev_zone == "N":
            return 1
        elif prev_zone == "D" and prev_owner == shooting_team_id:
            return 1
        elif prev_zone == "O" and prev_owner != shooting_team_id:
            return 1
        
        return 0

    def get_danger_zone(self, x, y):
        """
        Classify a shot as being in high, medium, or low danger zones. Zone calculation based on Zones defined by War-on-Ice.
        """

        x_abs = abs(x)
        y_abs = abs(y)

        if 69 <= x_abs <= 89 and y_abs <= 6:
            return "high"
         
        elif 69 <= x_abs <= 89 and 6 < y_abs <= 22:
            y_boundary = -0.8*x_abs + 77.2
            if y_abs <= y_boundary:
                return "med"
            else:
                 return "low"
            
        elif 44 <= x_abs < 69 and y_abs <= 22:
            return "med"
        else:
            return "low"


    def scrape_fenwick_shots(self, game_ids):
        """Takes in a list of game ids and scrapes the play-by-play for that game to create a DataFrame of Fenwick shot events."""
        rows = []

        for game_id in tqdm(game_ids, desc="Scraping games"):
            try:
                game_data = self.client.game_center.play_by_play(game_id=game_id)
            except Exception as e:
                return pd.DataFrame()

            home_id = game_data["homeTeam"]["id"]
            away_id = game_data["awayTeam"]["id"]
            pbp = game_data["plays"]

            for idx, play in enumerate(pbp):
                if play["typeDescKey"] not in {"missed-shot", "goal", "shot-on-goal"}:
                    continue

                try:
                    team_id = play["details"]["eventOwnerTeamId"]
                    home = int(team_id == home_id)
                    away = int(team_id == away_id)
                    home_side = play["homeTeamDefendingSide"]

                    if (home == 1 and play["situationCode"][0] == "0") or (away == 1 and play["situationCode"][3] == "0"):
                        continue

                    shooter_id = (
                        play["details"].get("scoringPlayerId")
                        if play["typeDescKey"] == "goal"
                        else play["details"].get("shootingPlayerId")
                    )
                    if shooter_id is None:
                        continue

                    shooter, shooter_stats = self.get_player_stats(shooter_id)

                    prev_play = pbp[idx - 1] if idx > 0 else None
                    rebound = self.is_rebound(play, prev_play)
                    rush = self.is_rush(play, prev_play)

                    x = play["details"].get("xCoord")
                    y = play["details"].get("yCoord")
                    if x is None or y is None:
                        continue
                    danger = self.get_danger_zone(x, y)

                    # Situations
                    home_skaters = int(play["situationCode"][2])
                    away_skaters = int(play["situationCode"][1])
                    shot_type = play["details"].get("shotType")
                    shot_class = play["typeDescKey"]
                    zone = play["details"].get("zoneCode")

                    # Goalie
                    goalie_id = play["details"].get("goalieInNetId")
                    if goalie_id:
                        goalie, goalie_stats = self.get_player_stats(goalie_id)
                    else:
                        goalie = None
                        goalie_stats = [None, None, None]

                    rows.append({
                    "game_id": game_id,
                    "team_id": team_id,
                    "home": home,
                    "home_def_side": home_side,
                    "last_play": prev_play["typeDescKey"] if prev_play else None,
                    "rebound": rebound,
                    "rush": rush,
                    "home_skaters": home_skaters,
                    "away_skaters": away_skaters,
                    "x_coord": x,
                    "y_coord": y,
                    "shooter_id": shooter_id,
                    "shooter": shooter,
                    "position": shooter_stats[0],
                    "shoots": shooter_stats[1],
                    "career_shooting_pct": shooter_stats[2],
                    "goalie_id": goalie_id,
                    "goalie": goalie,
                    "goalie_catches": goalie_stats[1],
                    "career_save_pct": goalie_stats[2],
                    "shot_type": shot_type,
                    "zone": zone,
                    "shot_class": shot_class,
                    "danger_zone": danger,
                })

                except Exception as e:
                    logging.warning(f"Error processing play {idx} in game {game_id}: {e}")
                    continue

        return pd.DataFrame(rows)