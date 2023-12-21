from pacti.contracts import PolyhedralIoContractCompound, PolyhedralIoContract
from utils.multiagent_utils import Robot
from itertools import combinations, product
from typing import List
import numpy as np
import itertools

def get_collision_contracts_list(robots: List[Robot]) -> PolyhedralIoContractCompound:
    """
    Contract ensuring no collision for all robots.

    Args:
        robots: list of robots.

    Returns:
        The contract that ensures no collision.
    """
    robotnames = []
    for robot in robots:
        robotnames.append(robot.name)

    combis = combinations(robots, 2)
    contracts: List[PolyhedralIoContractCompound] = []
    for combi in combis:
        contract = collision_contract_named(combi[0], combi[1])
        contracts.append(contract)

    return contracts


def collision_contract_named(robot_1: str, robot_2: str) -> PolyhedralIoContractCompound:
    """
    Contract ensuring no collision for a pair of robots.

    Args:
        robot_1: Name of robot 1.
        robot_2: Name of robot 2.

    Returns:
        The contract that ensures no collision.
    """
    inputvars = [
        "x_" + str(robot_1.name) + "_0",
        "y_" + str(robot_1.name) + "_0",
        "x_" + str(robot_2.name) + "_0",
        "y_" + str(robot_2.name) + "_0",
        "t_0",
        "current_distance",
    ]
    outputvars = [
        "x_" + str(robot_1.name) + "_1",
        "y_" + str(robot_1.name) + "_1",
        "x_" + str(robot_2.name) + "_1",
        "y_" + str(robot_2.name) + "_1",
        "t_1",
    ]

    contract = PolyhedralIoContractCompound.from_strings(
        input_vars=inputvars,
        output_vars=outputvars,
        assumptions=[["-current_distance <= -1"]],
        guarantees=[
            [
                "x_"
                + str(robot_1.name)
                + "_1 - x_"
                + str(robot_2.name)
                + "_1 + y_"
                + str(robot_1.name)
                + "_1 - y_"
                + str(robot_2.name)
                + "_1 <= -1"
            ],
            [
                "- x_"
                + str(robot_1.name)
                + "_1 + x_"
                + str(robot_2.name)
                + "_1 - y_"
                + str(robot_1.name)
                + "_1 + y_"
                + str(robot_2.name)
                + "_1 <= -1"
            ],
            [
                "x_"
                + str(robot_1.name)
                + "_1 - x_"
                + str(robot_2.name)
                + "_1 - y_"
                + str(robot_1.name)
                + "_1 + y_"
                + str(robot_2.name)
                + "_1 <= -1"
            ],
            [
                "- x_"
                + str(robot_1.name)
                + "_1 + x_"
                + str(robot_2.name)
                + "_1 + y_"
                + str(robot_1.name)
                + "_1 - y_"
                + str(robot_2.name)
                + "_1 <= -1"
            ],
        ],
    )
    return contract  # noqa: WPS331
