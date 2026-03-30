from typing import Annotated
from tool import Tool
import random


def roll(s: str):
    if "d" not in s:
        return int(s)
    s = s.strip()
    s_dice, s_faces = s.split("d")
    n_dice = int(s_dice.removeprefix("-"))
    n_faces = int(s_faces)
    result = sum(random.randint(1, n_faces) for _ in range(n_dice))
    if s_dice.startswith("-"):
        result = -result
    return result


class ToolRollDice(Tool):
    def __init__(self):
        super().__init__("roll_dice", "Roll dice.")

    def __call__(
        self,
        dice: Annotated[
            list,
            'array of dice to roll with possible bonuses or penalty. Example: ["2d6", "-2d4", "4"] means "roll two dice with six faces, then roll two dice with four faces and subtract from the result(minus in the start), then add four to the result as a bonus(no roll)',
        ],
    ):
        return sum(roll(x) for x in dice)
