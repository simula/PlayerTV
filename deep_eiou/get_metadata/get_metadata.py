import numpy as np
# !pip install nearest-colour
from nearest_colours import nearest_x11
import json

def find_nearest_two_colors(rgb_team_a, rgb_team_b, col_match_threshold=0.3, rgb_module=None):
    print(rgb_team_a)
    print(rgb_team_b)

    calculate_distance = lambda rgb_a, rgb_b: np.sqrt(np.sum((np.array(rgb_a) - np.array(rgb_b))**2)) / np.sqrt(3 * 255**2)
    nearest_color = lambda c: str(nearest_x11(color="#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2])),space='rgb')[0]).title()
    home_team, away_team = rgb_team_a[1], rgb_team_b[1] # default
    away_rgb = rgb_team_b[0] 

    if not team_colors_dict.get(rgb_team_a[1]):
        note = "Home team not found in the list."
        home_color = nearest_color(rgb_team_a[0])
    else:
      dist1 = calculate_distance(team_colors_dict[rgb_team_a[1]][1], rgb_team_a[0]) # Distance from point 1 to home team
      dist_1_b = calculate_distance(team_colors_dict[rgb_team_b[1]][1], rgb_team_a[0]) # Distance from point 2 to home team
      dist2 = calculate_distance(team_colors_dict[rgb_team_b[1]][1], rgb_team_b[0])
      dist_2_b = calculate_distance(team_colors_dict[rgb_team_a[1]][1], rgb_team_b[0])

      if dist1 < dist_1_b:
        home_team_idx = 0
        away_team_idx = 1
      else:
        home_team_idx = 1
        away_team_idx = 0
      if rgb_module:
        idx = rgb_module.predict_cluster(np.array(team_colors_dict[rgb_team_a[1]][1]))
        print("idx", idx)
        if idx != None and idx != -1:
          home_team_idx = idx
          away_team_idx = 1 - home_team_idx
    
      if (dist1 > col_match_threshold and dist2 > col_match_threshold) or not team_colors_dict.get(rgb_team_a[1]):
          try:
            home_color = nearest_color(rgb_team_a[0])
          except:
            home_color = "" # so directly name given color
          if not team_colors_dict.get(rgb_team_a[1]):
            note = "Home team not found in the list."
          else:
            note =  "Both colors are far from the two team's home color"+ f" dist1: {dist1}, dist2: {dist2}."
      else:
          if dist1 < dist2:
              home_color = team_colors_dict[rgb_team_a[1]][0] # rgb1 was home color
          else:
              # rgb2 was similar to home color, and need to update away_rgb as well
              home_color, away_rgb = team_colors_dict[rgb_team_b[1]][0], rgb_team_a[0]
          note = f"dist point a team a: {dist1}, dist point b team a: {dist_1_b}, dist point b team b: {dist2}, dist point a team b: {dist_2_b}."
    try:
      away_color = nearest_color(away_rgb)
    except:
      away_color = ""
    return {
        "home_team": home_team, "home_color": home_color, "away_team": away_team, "away_color": away_color,
        "note": note, "home_team_idx": home_team_idx, "away_team_idx": away_team_idx
    }



# resove color with TeamID
import requests
def resolve_color(teamID, rgb_team_a, rgb_team_b, col_match_threshold=0.3, api_info=None, rgb_module=None):
  assert int(teamID)> 0
  if not api_info:
    api= "https://api.forzify.com/eliteserien/game/"+ str(teamID) +"/players"
    response = requests.get(api)
    data = response.json()
    home_team, away_team = data['home_team']['name'], data['visiting_team']['name']
    home_team_player=[{
        'name': player['name'],
        'shirt_number': player['shirt_number'],
    } for player in data['home_team']['players']]
    away_team_player=[{
        'name': player['name'],
        'shirt_number': player['shirt_number'],
    } for player in data['visiting_team']['players']]
  else:
     home_team = api_info["home_team"]
     away_team = api_info["away_team"]
     home_team_player = api_info["home_team_player"]
     away_team_player = api_info["away_team_player"]

  results = find_nearest_two_colors(
      [rgb_team_a, home_team],
      [rgb_team_b, away_team],
      col_match_threshold=col_match_threshold,
      rgb_module = rgb_module
  )
  results.update({
    "home_team_player": home_team_player,
    "away_team_player": away_team_player
  })
  return results

def get_video_url(event):
    api= "https://api.forzify.com/eliteserien/event/"+str(event)
    response = requests.get(api)
    data = response.json()
    return data["playlist"]["video_url"]


team_colors_dict = {'Bodø/Glimt': ['Yellow/Black', ([127.5, 127.5,   0. ])],
 'Brann': ['Red/White', ([255. , 127.5, 127.5])],
 'HamKam': ['White/Green', ([127.5, 191.5, 127.5])],
 'Haugesund': ['White/blue', ([127.5, 127.5, 255. ])],
 'Lillestrøm': ['Yellow/Black', ([127.5, 127.5,   0. ])],
 'Molde': ['Blue/white', ([127.5, 127.5, 255. ])],
 'Odd': ['White/black', ([127.5, 127.5, 127.5])],
 'Rosenborg': ['White/Green', ([127.5, 191.5, 127.5])],
 'Sandefjord Fotball': ['Navy blue/red', ([127.5,   0. ,  64. ])],
 'Sarpsborg 08': ['Blue/white', ([127.5, 127.5, 255. ])],
 'Stabæk': ['Blue-black/white', ([127.5, 127.5, 197. ])],
 'Strømsgodset': ['Navy blue/white', ([127.5, 127.5, 191.5])],
 'Tromsø': ['Red/White', ([255. , 127.5, 127.5])],
 'Viking': ['Navy blue/white', ([127.5, 127.5, 191.5])],
 'Vålerenga': ['Blue/White', ([127.5, 127.5, 255. ])],
 'Aalesund': ['Orange/white', ([255. , 210. , 127.5])]}
#### CODE TO CREATE team_colors_dict FROM MEHDI's sheets
# import pandas as pd
# import numpy as np
# data =pd.ExcelFile("https://docs.google.com/spreadsheets/d/e/2PACX-1vSJyrC3F4RLRRBJcWD27aKiTBWKkVtip8nAapxor4E5IKTIobQSm9eOHPl09aUWCVE2SAThJIG_MqWU/pub?output=xlsx")

# sheets = list(set(data.sheet_names)-set('README'))
# sheets_dict ={}
# for e in sheets:
#   sheets_dict[e] = data.parse(e)
#   df_list = []
# for sheet_name, df in sheets_dict.items():
#     df['Sheet'] = sheet_name
#     df_list.append(df)
# combined_df = pd.concat(df_list, ignore_index=True)
# combined_df = combined_df.dropna(subset=['Kit color'])
# combined_df = combined_df.loc[:, ~combined_df.columns.str.startswith('Unnamed')]
# combined_df = combined_df[~combined_df['League'].str.startswith('http')]
# combined_df["Sheet"] = combined_df["Sheet"].map(lambda x: x.split("-")[0])
# combined_df

# all_colors = combined_df['Kit color'].map(lambda x: x.lower().split('/')).explode().unique()

# color_rgb_ = {
#     'white': (255, 255, 255),
#     'green': (0, 128, 0),
#     'red': (255, 0, 0),
#     'blue-black': (0, 0, 139),  # Assuming a dark blue
#     'yellow': (255, 255, 0),
#     'black': (0, 0, 0),
#     'blue': (0, 0, 255),
#     'navy blue': (0, 0, 128),
#     'orange': (255, 165, 0),
# }
# print(set(all_colors) - set(color_rgb_.keys()))
# assert set(color_rgb_.keys()) == set(all_colors) # all color has mapped

# team_colors_count =combined_df.groupby(['Team', 'Kit color']).size().reset_index(name='count')
# team_colors_count['rgb'] = team_colors_count['Kit color'].map(lambda x: list(np.array([color_rgb_.get(e.lower()) for e in x.split('/')]).mean(axis=0)))

# team_colors_dict = team_colors_count.groupby('Team').apply(
#     lambda x: x[['Kit color', 'rgb']].values.tolist()[0]
# ).to_dict()
# team_colors_dict
#### END CODE TO CREATE team_colors_dict FROM MEHDI's sheets




# Example 1
#find_nearest_two_colors( 
#    [(0, 0, 0), "Brann"],
#    [(255, 0, 0), "Ålesund"],
#    col_match_threshold=0.7 # small means high accuracy
#   )
# Output: {'home_team': 'Ålesund',
#  'home_color': 'Orange/white',
#  'away_team': 'Brann',
#  'away_color': 'Black',
#  'note': 'dist1: 0.7071067811865475, dist2: 0.5562375667648738.'}


# Example 2 with TeamID
#resolve_color(
#    3783 ,
#   (162, 180, 75) ,
#    (138, 172, 126), 
#    col_match_threshold=0.9)

# Output: {'home_team': 'Molde',
#  'home_color': 'Blue/white',
#  'away_team': 'Vålerenga',
#  'away_color': 'Black',
#  'note': 'dist1: 0.7071067811865475, dist2: 0.7071067811865475.',
#  'home_team_player':[]
#  'away_team_player':[]}

