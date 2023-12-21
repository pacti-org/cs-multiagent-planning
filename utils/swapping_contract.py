from pacti.contracts import PolyhedralIoContract, PolyhedralIoContractCompound
from utils.multiagent_utils import Robot
from itertools import combinations, product
from typing import List

def get_compound_swapping_contract(robot_1, robot_2):
    """
    Contract ensuring no swapping for a pair of robots.

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
    ]
    outputvars = [
        "x_" + str(robot_1.name) + "_1",
        "y_" + str(robot_1.name) + "_1",
        "x_" + str(robot_2.name) + "_1",
        "y_" + str(robot_2.name) + "_1",
    ]

    contract = PolyhedralIoContractCompound.from_strings(
        input_vars=inputvars,
        output_vars=outputvars,
        assumptions=[[]],
        guarantees=[
            ["x_"+robot_1.name+ "_1 - x_" +robot_2.name+ "_0 + y_" +robot_1.name+ "_1 - y_" +robot_2.name+ "_0 >= 1"],
            ["x_"+robot_1.name+ "_1 - x_" +robot_2.name+ "_0 + y_" +robot_1.name+ "_1 - y_" +robot_2.name+ "_0 <= -1"],
            ["x_"+robot_2.name+ "_1 - x_" +robot_1.name+ "_0 + y_" +robot_2.name+ "_1 - y_" +robot_1.name+ "_0 >= 1"],
            ["x_"+robot_2.name+ "_1 - x_" +robot_1.name+ "_0 + y_" +robot_2.name+ "_1 - y_" +robot_1.name+ "_0 <= -1"],
            ]
        )
    return contract  # noqa: WPS331

def get_swapping_contracts_list(robots: List[Robot]) -> PolyhedralIoContractCompound:
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
        contract = get_compound_swapping_contract(combi[0], combi[1])
        contracts.append(contract)

    return contracts  # noqa: WPS331
