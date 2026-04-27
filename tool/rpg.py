from typing import Annotated, Any
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
        return {"dice": sum(roll(x) for x in dice)}


class ToolRollCheck(Tool):
    def __init__(self):
        super().__init__(
            "roll_check",
            'Roll 3d6 with bonus against target number and get the result. Depending on margin (dice roll - target) it will be described as LEGENDARY(>=6), STRONG(>=3), SUCCESS(>=0), WEAK FAILURE(>= -3), STRONG FAILURE(>=-6), DISASTER(< -6). Each outcome has "degree" of success (from 3 for LEGENDARY to -3 for DISASTER)',
        )

    def __call__(
        self,
        brief_description: Annotated[
            str, "Brief one-sentence description of the dice roll"
        ],
        bonus: Annotated[
            int, "Bonus(>0) or penalty(<0) added to the roll. (0 if no bonus)"
        ],
        target: Annotated[int, "Target number"],
    ):
        dice = roll("3d6") + bonus
        margin = dice - target
        thresholds = [
            (6, "LEGENDARY", 3),
            (3, "STRONG", 2),
            (0, "SUCCESS", 1),
            (-3, "WEAK FAILURE", -1),
            (-6, "STRONG FAILURE", -2),
            (-100, "DISASTER", -3),
        ]

        for n, desc, degree in thresholds:
            if margin >= n:
                return {
                    "description": brief_description,
                    "dice": dice,
                    "result": desc,
                    "degree": degree,
                    "margin": margin,
                }
        raise ValueError("Should never be reached")


class ToolRollVersus(Tool):
    def __init__(self):
        super().__init__(
            "roll_vs",
            "Roll 3d6 with bonuses for two parties of LHS and RHS and describe the result and decide the winner. Will return tie or the winner and margin of victory(always >=0)",
        )

    def __call__(
        self,
        lhs_description: Annotated[
            str, 'One sentence description of LHS, example: "Frodo: aims at Gollm"'
        ],
        lhs_bonus: Annotated[int, "Bonus(>0) or penalty(<0) added to LHS roll"],
        rhs_description: Annotated[
            str, 'One sentence description of RHS, example: "Gollms: dodges the attack"'
        ],
        rhs_bonus: Annotated[int, "Bonus(>0) or penalty(<0) added to RHS roll"],
    ):
        lhs = roll("3d6") + lhs_bonus
        rhs = roll("3d6") + rhs_bonus

        res: dict[str, Any] = {"lhs": lhs, "rhs": rhs, "margin": abs(lhs - rhs)}
        if lhs >= rhs:
            res["winner"] = lhs_description
        elif rhs >= lhs:
            res["winner"] = rhs_description
        else:
            return {"tie": True, "roll": lhs}
        return res
