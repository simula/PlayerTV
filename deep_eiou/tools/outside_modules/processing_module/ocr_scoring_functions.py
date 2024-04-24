import numpy as np

def highest_in_player_list(txt, box, conf, accumulator, crop, player_list=None):
    if player_list:
        for entry in player_list:
            kit_num = entry["shirt_number"]
            txt = int(txt)
            if txt == kit_num:
                return (conf*accumulator*10), entry["name"]
    return conf*accumulator, None