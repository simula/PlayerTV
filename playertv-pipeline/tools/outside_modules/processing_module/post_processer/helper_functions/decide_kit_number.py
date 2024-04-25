from typing import Union


def highest_in_player_list(player_list: list[dict], txt: Union[str, int], conf: float, iou: float, accumulator: int):
    try:
        txt = int(txt)
        for player in player_list:
            if player.get("shirt_number") == txt:
                alt_txt = f"{player['name']} ({player['shirt_number']})"
                score = (conf * 10 * accumulator) * (1-iou**2)
                return score, alt_txt
            
    except:
        return conf, None
    return conf, None
