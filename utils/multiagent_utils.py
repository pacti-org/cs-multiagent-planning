"""Utility functions for the multiagent coordination case study."""
import random
from itertools import combinations, product
from typing import List

import numpy as np
import time
import json

from pacti.iocontract import Var
from pacti.contracts import PolyhedralIoContractCompound

class Coord:
    """
    Robot coordinate class.

    Allows storing the x and y coordinate.
    """

    def __init__(self, pos: tuple[int, int]):  # noqa: D107
        self.x = pos[0]
        self.y = pos[1]
        self.xy = pos


class Robot:
    """
    Robot class.

    Stores the name, position and goal of the robot.
    """

    def __init__(self, name: str, pos: tuple[int, int], goal: tuple[int, int]):  # noqa: D107
        self.name = name
        self.pos = Coord(pos)
        self.goal = Coord(goal)
        self.at_goal = False

    def move(self, new_pos: tuple[int, int]) -> None:  # noqa: D102
        self.pos = Coord(new_pos)
        if distance([(self.pos.x, self.pos.y)],[self.goal]) == 0:
            self.at_goal = True
        else:
            self.at_goal = False


def distance(candidate: List[tuple[int, int]], goal: List[Coord]) -> int:
    """
    Distance measure from next robot positions to desired position.

    Args:
        candidate: list of the next positions for the robots.
        goal: list of goals corresponding to the robots.

    Returns:
        The cumulated distance from the candidate next position to the goal.
    """
    distance: int = 0  # noqa: WPS442
    for i, move in enumerate(candidate):
        distance = distance + np.abs(move[0] - goal[i].x) + np.abs(move[1] - goal[i].y)  # noqa: WPS442

    return distance


def indiv_distance(move: tuple[int, int], goal: Coord) -> int:
    """
    Distance measure from current robot position to desired position.

    Args:
        move: Next robot position.
        goal: Robot goal position.

    Returns:
        The distance from the robot's next position to the goal.
    """
    distance: int = np.abs(move[0] - goal.x) + np.abs(move[1] - goal.y)  # noqa: WPS442
    return distance  # noqa: WPS331


def strategy(move_candidates: List[List[tuple[int, int]]], goal: List[Coord]) -> List[tuple[int, int]]:  # noqa: WPS234
    """
    Choosing the next move according to a strategy.

    Args:
        move_candidates: list of next robot positions.
        goal: list of goals corresponding to the robots.

    Returns:
        The chosen move.
    """
    min_dist: dict[int, List[List[tuple[int, int]]]] = {}  # noqa: WPS234
    for candidate in move_candidates:
        candidate_list: List[List[tuple[int, int]]] = []  # noqa: WPS234
        dist = distance(candidate, goal)

        if dist in min_dist.keys():
            for entry in min_dist[dist]:
                candidate_list.append(entry)

        candidate_list.append(candidate)
        min_dist.update({dist: candidate_list})

    move = random.choice(min_dist[min(sorted(min_dist.keys()))])  # noqa: S311
    return move  # noqa: WPS331


def strategy_multiple_simple(  # noqa: WPS234
    move_candidates: List[List[tuple[int, int]]], goal: List[Coord]
) -> List[tuple[int, int]]:
    """
    Choosing the next move for multiple robots according to a simple strategy.

    Args:
        move_candidates: list of next robot positions.
        goal: list of goals corresponding to the robots.

    Returns:
        The chosen move.
    """
    min_dist: dict[int, list] = {}
    for candidate in move_candidates:
        candidate_list = []
        dist = distance(candidate, goal)

        if dist in min_dist.keys():
            for entry in min_dist[dist]:
                candidate_list.append(entry)

        candidate_list.append(candidate)
        min_dist.update({dist: candidate_list})

    if random.random() < 0.2:  # noqa: S311,WPS459,WPS432
        move = random.choice(move_candidates)  # noqa: S311
    else:
        move = random.choice(min_dist[min(sorted(min_dist.keys()))])  # noqa: S311

    return move


def strategy_multiple(  # noqa: WPS231,WPS234
    move_candidates: List[List[tuple[int, int]]],
    goal: List[Coord],
    cur_pos: List[tuple[int, int]],
    last_pos: List[tuple[int, int]],
) -> List[tuple[int, int]]:
    """
    Choosing the next move for multiple robots according to a strategy.

    Args:
        move_candidates: list of next robot positions.
        goal: list of goals corresponding to the robots.
        cur_pos: Current position.
        last_pos: Last position.

    Returns:
        The chosen move.
    """
    min_dist: dict[int, list] = {}
    for candidate in move_candidates:
        candidate_list = []
        dist = distance(candidate, goal)

        if dist in min_dist.keys():
            for entry in min_dist[dist]:
                candidate_list.append(entry)
        if candidate not in candidate_list:
            candidate_list.append(candidate)
        min_dist.update({dist: candidate_list})  # noqa: S311

    best_move = random.choice(min_dist[min(sorted(min_dist.keys()))])  # noqa: S311

    best_dist = distance(best_move, goal)
    cur_dist = distance(cur_pos, goal)

    moves: List[List[tuple[int, int]]] = []  # noqa: WPS234
    chosen_move: List[tuple[int, int]] = []  # noqa: WPS234

    if best_dist < cur_dist:
        move_options: List[List[tuple[int, int]]] = []  # noqa: WPS234
        for distances in min_dist.keys():
            if distances < cur_dist:
                move_options = move_options + min_dist[distances]
        # prune options where robots leave their goal
        for move in move_options:
            move_ok = True
            for i, _value in enumerate(cur_pos):  # keep robots at goal at goal
                if indiv_distance(move[i], goal[i]) < indiv_distance(move[i], goal[i]):
                    move_ok = False
            if move_ok and move not in moves:
                moves = moves + [move]  # noqa: S311

        if random.random() < 0.1:  # noqa: S311,WPS459,WPS432
            chosen_move = random.choice(min_dist[min(sorted(min_dist.keys()))])  # noqa: S311
        else:  # take a random good move
            chosen_move = random.choice(moves)  # noqa: S311
    # check that other moves are possible
    elif best_dist == cur_dist:
        for move in move_candidates:
            move_ok = True
            if move == cur_pos:  # remove staying in the same position
                move_ok = False
            if move == last_pos:  # remove going back to previous position
                move_ok = False
            for i, _val in enumerate(cur_pos):  # keep robots at goal at goal
                if indiv_distance(cur_pos[i], goal[i]) == 0 and indiv_distance(move[i], goal[i]) != 0:
                    move_ok = False
            if move_ok and move not in moves:  # add move to possible moves
                moves = moves + [move]

        chosen_move = random.choice(moves)  # noqa: S311

    return chosen_move


def find_move_candidates_three(  # noqa: WPS231
    grid_n: int, grid_m: int, robots: List[Robot], T_0: int, contract: PolyhedralIoContractCompound  # noqa: N803
) -> tuple[list, int]:
    """
    Evaluate the contracts for possible next positions of the robots to find allowed moves.

    Args:
        grid_n: Grid dimension n.
        grid_m: Grid dimension m.
        robots: list of robots.
        T_0: Current time step.
        contract: The contract that the next positions must satisfy.

    Returns:
        A list of possible next positions and the next timestep.
    """
    x_A_0 = Var("x_A_0")  # noqa: N806
    y_A_0 = Var("y_A_0")  # noqa: N806
    x_B_0 = Var("x_B_0")  # noqa: N806
    y_B_0 = Var("y_B_0")  # noqa: N806
    x_C_0 = Var("x_C_0")  # noqa: N806
    y_C_0 = Var("y_C_0")  # noqa: N806
    current_distance = Var("current_distance")
    t_0 = Var("t_0")
    t_1 = Var("t_1")
    x_A_1 = Var("x_A_1")  # noqa: N806
    y_A_1 = Var("y_A_1")  # noqa: N806
    x_B_1 = Var("x_B_1")  # noqa: N806
    y_B_1 = Var("y_B_1")  # noqa: N806
    x_C_1 = Var("x_C_1")  # noqa: N806
    y_C_1 = Var("y_C_1")  # noqa: N806
    delta_x_A_B = Var("delta_x_A_B")  # noqa: N806
    delta_y_A_B = Var("delta_y_A_B")  # noqa: N806
    delta_x_A_C = Var("delta_x_A_C")  # noqa: N806
    delta_y_A_C = Var("delta_y_A_C")  # noqa: N806
    delta_x_B_C = Var("delta_x_B_C")  # noqa: N806
    delta_y_B_C = Var("delta_y_B_C")  # noqa: N806

    X_A_0 = robots[0].pos.x  # noqa: N806
    Y_A_0 = robots[0].pos.y  # noqa: N806
    X_B_0 = robots[1].pos.x  # noqa: N806
    Y_B_0 = robots[1].pos.y  # noqa: N806
    X_C_0 = robots[2].pos.x  # noqa: N806
    Y_C_0 = robots[2].pos.y  # noqa: N806
    current_distance_1 = np.abs(X_A_0 - X_B_0) + np.abs(Y_A_0 - Y_B_0)
    current_distance_2 = np.abs(X_B_0 - X_C_0) + np.abs(Y_B_0 - Y_C_0)
    current_distance_3 = np.abs(X_A_0 - X_C_0) + np.abs(Y_A_0 - Y_C_0)
    cur_dist = min(current_distance_1, current_distance_2, current_distance_3)
    T_1 = T_0 + 1  # noqa: N806

    # find possible [(x,y),(x,y),(x,y)] options for robots
    possible_sol: List[List[tuple[int, int]]] = []  # noqa: WPS234
    for x_a in list({max(X_A_0 - 1, 0), X_A_0, min(X_A_0 + 1, grid_n)}):
        for y_a in list({max(Y_A_0 - 1, 0), Y_A_0, min(Y_A_0 + 1, grid_m)}):
            for x_b in list({max(X_B_0 - 1, 0), X_B_0, min(X_B_0 + 1, grid_n)}):
                for y_b in list({max(Y_B_0 - 1, 0), Y_B_0, min(Y_B_0 + 1, grid_m)}):
                    for x_c in list({max(X_C_0 - 1, 0), X_C_0, min(X_C_0 + 1, grid_n)}):
                        for y_c in list({max(Y_C_0 - 1, 0), Y_C_0, min(Y_C_0 + 1, grid_m)}):
                            del_x_A_B = (x_a - x_b) * (X_A_0 - X_B_0)  # noqa: N806
                            del_y_A_B = (y_a - y_b) * (Y_A_0 - Y_B_0)  # noqa: N806
                            del_x_A_C = (x_a - x_c) * (X_A_0 - X_C_0)  # noqa: N806
                            del_y_A_C = (y_a - y_c) * (Y_A_0 - Y_C_0)  # noqa: N806
                            del_x_B_C = (x_b - x_c) * (X_B_0 - X_C_0)  # noqa: N806
                            del_y_B_C = (y_b - y_c) * (Y_B_0 - Y_C_0)  # noqa: N806

                            var_dict = {
                                x_A_0: X_A_0,
                                y_A_0: Y_A_0,
                                x_B_0: X_B_0,
                                y_B_0: Y_B_0,
                                x_C_0: X_C_0,
                                y_C_0: Y_C_0,
                                current_distance: cur_dist,
                                t_0: T_0,
                                t_1: T_1,
                                x_A_1: x_a,
                                y_A_1: y_a,
                                x_B_1: x_b,
                                y_B_1: y_b,
                                x_C_1: x_c,
                                y_C_1: y_c,
                                delta_x_A_B: del_x_A_B,
                                delta_y_A_B: del_y_A_B,
                                delta_x_A_C: del_x_A_C,
                                delta_y_A_C: del_y_A_C,
                                delta_x_B_C: del_x_B_C,
                                delta_y_B_C: del_y_B_C,
                            }

                            if contract.a.contains_behavior(var_dict) and contract.g.contains_behavior(var_dict):
                                possible_sol.append([(x_a, y_a), (x_b, y_b), (x_c, y_c)])

    return possible_sol, T_1


def find_move_candidates_general(  # noqa: WPS231
    grid_n: int, grid_m: int, robots: List[Robot], T_0: int, contract: PolyhedralIoContractCompound  # noqa: N803
) -> tuple[list, int]:
    """
    Evaluate the contracts for possible next positions of the robots to find allowed moves.

    Args:
        grid_n: Grid dimension n.
        grid_m: Grid dimension m.
        robots: list of robots.
        T_0: Current time step.
        contract: The contract that the next positions must satisfy.

    Returns:
        A list of possible next positions and the next timestep.
    """
    robotnames = []
    var_dict = {}
    for robot in robots:
        robotnames.append(robot.name)
        var_dict.update({Var("x_"+str(robot.name)+"_0"): robot.pos.x})
        var_dict.update({Var("y_"+str(robot.name)+"_0"): robot.pos.y})

    robotcombis = combinations(robots, 2)
    cur_dist = 99
    for combi in list(robotcombis):
        distance_btw_robots = np.abs(combi[0].pos.x - combi[1].pos.x) + np.abs(combi[0].pos.y - combi[1].pos.y)
        cur_dist = min(cur_dist, distance_btw_robots)
    T_1 = T_0 + 1  # noqa: N806
    var_dict.update({Var("t_0"): T_0})
    var_dict.update({Var("t_1"): T_1})
    var_dict.update({Var("current_distance"): cur_dist})

    # current distance to the goal
    to_goal = distance([[robot.pos.x, robot.pos.y] for robot in robots], [robot.goal for robot in robots])
    var_dict.update({Var("cur_dist_to_goal"): to_goal})

    # next possible positions
    x_list = list({max(robots[0].pos.x - 1, 0), robots[0].pos.x, min(robots[0].pos.x + 1, grid_n)})
    y_list = list({max(robots[0].pos.y - 1, 0), robots[0].pos.y, min(robots[0].pos.y + 1, grid_m)})
    next_possible_positions = []
    for x in x_list:
        for y in y_list:
            next_possible_positions.append([(x,y)])

    for robot in robots[1:]:
        x_list = list({max(robot.pos.x - 1, 0), robot.pos.x, min(robot.pos.x + 1, grid_n)})
        y_list = list({max(robot.pos.y - 1, 0), robot.pos.y, min(robot.pos.y + 1, grid_m)})
        pos = []
        for x in x_list:
            if x == robot.pos.x:
                for y in y_list:
                    pos.append([(x,y)])
            else:
                y = robot.pos.y
                pos.append([(x,y)])

        # removing collision states
        next_list = []
        for pos_position in next_possible_positions:
            for new_robot_pos in pos:
                if new_robot_pos[0] not in pos_position:
                    entry = pos_position + new_robot_pos
                    next_list.append(entry)

        next_possible_positions = next_list

    # remove duplicates (shouldn't have any)
    import itertools
    next_possible_positions.sort()
    next_possible_positions = [z for z,_ in itertools.groupby(next_possible_positions)]

    print('total possible positions are {}'.format(len(next_possible_positions)))

    # remove a random element from list
    while len(next_possible_positions) > 0:
        possible_position = random.choice(next_possible_positions)
        next_possible_positions.remove(possible_position)

        pos_dict = {}
        for i,robot in enumerate(robots):
            pos_dict.update({robot.name: possible_position[i]})
            var_dict.update({Var("x_"+str(robot.name)+"_1"): possible_position[i][0]})
            var_dict.update({Var("y_"+str(robot.name)+"_1"): possible_position[i][1]})

        robotcombis = combinations(robots, 2)
        for combi in list(robotcombis):
            del_x_str = "delta_x_" + str(combi[0].name) + "_" + str(str(combi[1].name))
            del_y_str = "delta_y_" + str(combi[0].name) + "_" + str(str(combi[1].name))
            del_x = (pos_dict[combi[0].name][0] - pos_dict[combi[1].name][0] ) * (combi[0].pos.x - combi[1].pos.x)
            del_y = (pos_dict[combi[0].name][1] - pos_dict[combi[1].name][1] ) * (combi[0].pos.y - combi[1].pos.y)
            var_dict.update({Var(del_x_str): del_x})
            var_dict.update({Var(del_y_str): del_y})

        # next distance to the goal
        next_to_goal = distance([[possible_position[i][0], possible_position[i][1]] for i in range(len(possible_position))], [robot.goal for robot in robots])
        var_dict.update({Var("next_dist_to_goal"): next_to_goal})

        ti = time.time()
        if contract.a.contains_behavior(var_dict) and contract.g.contains_behavior(var_dict):
            tf = time.time()
            print('{0} took {1}'.format(var_dict, tf-ti))
            return possible_position, T_1
        tf = time.time()
        print('{0} took {1}'.format(var_dict, tf-ti))

        str_var_dict = {var.name: var_dict[var] for var in var_dict.keys()}
        with open('test_behavior.json', 'w') as fp:
            json.dump(str_var_dict, fp)


    return [], T_1

def find_move_candidates_check_all(  # noqa: WPS231
    grid_n: int, grid_m: int, robots: List[Robot], T_0: int, contract: PolyhedralIoContractCompound  # noqa: N803
) -> tuple[list, int]:
    """
    Evaluate the contracts for possible next positions of the robots to find allowed moves.

    Args:
        grid_n: Grid dimension n.
        grid_m: Grid dimension m.
        robots: list of robots.
        T_0: Current time step.
        contract: The contract that the next positions must satisfy.

    Returns:
        A list of possible next positions and the next timestep.
    """
    robotnames = []
    var_dict = {}
    for robot in robots:
        robotnames.append(robot.name)
        var_dict.update({Var("x_"+str(robot.name)+"_0"): robot.pos.x})
        var_dict.update({Var("y_"+str(robot.name)+"_0"): robot.pos.y})

    robotcombis = combinations(robots, 2)
    cur_dist = 99
    for combi in list(robotcombis):
        distance_btw_robots = np.abs(combi[0].pos.x - combi[1].pos.x) + np.abs(combi[0].pos.y - combi[1].pos.y)
        cur_dist = min(cur_dist, distance_btw_robots)
    T_1 = T_0 + 1  # noqa: N806
    var_dict.update({Var("t_0"): T_0})
    var_dict.update({Var("t_1"): T_1})
    var_dict.update({Var("current_distance"): cur_dist})

    # current distance to the goal
    to_goal = distance([[robot.pos.x, robot.pos.y] for robot in robots], [robot.goal for robot in robots])
    var_dict.update({Var("cur_dist_to_goal"): to_goal})

    # next possible positions
    x_list = list({max(robots[0].pos.x - 1, 0), robots[0].pos.x, min(robots[0].pos.x + 1, grid_n)})
    y_list = list({max(robots[0].pos.y - 1, 0), robots[0].pos.y, min(robots[0].pos.y + 1, grid_m)})
    next_possible_positions = []
    for x in x_list:
        for y in y_list:
            next_possible_positions.append([(x,y)])

    for robot in robots[1:]:
        x_list = list({max(robot.pos.x - 1, 0), robot.pos.x, min(robot.pos.x + 1, grid_n)})
        y_list = list({max(robot.pos.y - 1, 0), robot.pos.y, min(robot.pos.y + 1, grid_m)})
        pos = []
        for x in x_list:
            for y in y_list:
                pos.append([(x,y)])

        next_list = []
        for pos_position in next_possible_positions:
            for new_robot_pos in pos:
                if new_robot_pos[0] not in pos_position:
                    entry = pos_position + new_robot_pos
                    next_list.append(entry)

        next_possible_positions = next_list

    possible_sol = []
    for possible_position in next_possible_positions:
        pos_dict = {}
        for i,robot in enumerate(robots):
            pos_dict.update({robot.name: possible_position[i]})
            var_dict.update({Var("x_"+str(robot.name)+"_1"): possible_position[i][0]})
            var_dict.update({Var("y_"+str(robot.name)+"_1"): possible_position[i][1]})

        robotcombis = combinations(robots, 2)
        for combi in list(robotcombis):
            del_x_str = "delta_x_" + str(combi[0].name) + "_" + str(str(combi[1].name))
            del_y_str = "delta_y_" + str(combi[0].name) + "_" + str(str(combi[1].name))
            del_x = (pos_dict[combi[0].name][0] - pos_dict[combi[1].name][0] ) * (combi[0].pos.x - combi[1].pos.x)
            del_y = (pos_dict[combi[0].name][1] - pos_dict[combi[1].name][1] ) * (combi[0].pos.y - combi[1].pos.y)
            var_dict.update({Var(del_x_str): del_x})
            var_dict.update({Var(del_y_str): del_y})

        # current distance to the goal
        next_to_goal = distance([[possible_position[i][0], possible_position[i][1]] for i in range(len(possible_position))], [robot.goal for robot in robots])
        var_dict.update({Var("next_dist_to_goal"): next_to_goal})

        if contract.a.contains_behavior(var_dict) and contract.g.contains_behavior(var_dict):
            possible_sol.append(possible_position)

    return possible_sol, T_1


def robots_move(robots: List[Robot], move: List[tuple[int, int]]) -> None:
    """
    Apply next move and update positions.

    Args:
        robots: list of robots.
        move: Next move.
    """
    for i, _robot in enumerate(robots):
        robots[i].move(move[i])


def get_swapping_contract(robots: List[Robot]) -> PolyhedralIoContractCompound:
    """
    Contract ensuring no swapping conflicts for all robots.

    Args:
        robots: list of robots.

    Returns:
        The contract that ensures no swapping conflicts.
    """
    robotnames = []
    for robot in robots:
        robotnames.append(robot.name)

    combis = combinations(robotnames, 2)

    inputvars = ["current_distance"]
    delta_pairs = []
    for combi in combis:
        del_x_str = "delta_x_" + str(combi[0]) + "_" + str(str(combi[1]))
        del_y_str = "delta_y_" + str(combi[0]) + "_" + str(str(combi[1]))
        inputvars = inputvars + [del_x_str, del_y_str]
        delta_pairs.append((del_x_str, del_y_str))
    outputvars: list = []

    contract = PolyhedralIoContractCompound.from_strings(
        input_vars=inputvars,
        output_vars=outputvars,
        assumptions=[["-current_distance <= -1"]],
        guarantees=[[f"-{delta[0]}-{delta[1]} <= -1" for delta in delta_pairs]],
    )

    return contract  # noqa: WPS331


def get_collision_contract(robots: List[Robot]) -> PolyhedralIoContractCompound:
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

    combis = combinations(robotnames, 2)
    contracts: List[PolyhedralIoContractCompound] = []
    for combi in combis:
        contract = collision_contract_named(combi[0], combi[1])
        contracts.append(contract)

    c_collison = contracts[0]
    for contract in contracts[1:]:
        c_collison = c_collison.merge(contract)

    return c_collison


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
        "x_" + str(robot_1) + "_0",
        "y_" + str(robot_1) + "_0",
        "x_" + str(robot_2) + "_0",
        "y_" + str(robot_2) + "_0",
        "t_0",
        "current_distance",
    ]
    outputvars = [
        "x_" + str(robot_1) + "_1",
        "y_" + str(robot_1) + "_1",
        "x_" + str(robot_2) + "_1",
        "y_" + str(robot_2) + "_1",
        "t_1",
    ]

    contract = PolyhedralIoContractCompound.from_strings(
        input_vars=inputvars,
        output_vars=outputvars,
        assumptions=[["-current_distance <= -1"]],
        guarantees=[
            [
                "x_"
                + str(robot_1)
                + "_1 - x_"
                + str(robot_2)
                + "_1 + y_"
                + str(robot_1)
                + "_1 - y_"
                + str(robot_2)
                + "_1 <= -1"
            ],
            [
                "- x_"
                + str(robot_1)
                + "_1 + x_"
                + str(robot_2)
                + "_1 - y_"
                + str(robot_1)
                + "_1 + y_"
                + str(robot_2)
                + "_1 <= -1"
            ],
            [
                "x_"
                + str(robot_1)
                + "_1 - x_"
                + str(robot_2)
                + "_1 - y_"
                + str(robot_1)
                + "_1 + y_"
                + str(robot_2)
                + "_1 <= -1"
            ],
            [
                "- x_"
                + str(robot_1)
                + "_1 + x_"
                + str(robot_2)
                + "_1 + y_"
                + str(robot_1)
                + "_1 - y_"
                + str(robot_2)
                + "_1 <= -1"
            ],
        ],
    )
    return contract  # noqa: WPS331


def find_move_candidates_separate(  # noqa: WPS231
    grid_n: int, grid_m: int, robots: List[Robot], T_0: int, contract: PolyhedralIoContractCompound, # noqa: N803
    contractlist_1 : List[PolyhedralIoContractCompound], contractlist_2 : List[PolyhedralIoContractCompound] ) -> tuple[list, int]:
    """
    Evaluate the contracts for possible next positions of the robots to find allowed moves.

    Args:
        grid_n: Grid dimension n.
        grid_m: Grid dimension m.
        robots: list of robots.
        T_0: Current time step.
        contract: The contract that the next positions must satisfy.

    Returns:
        A list of possible next positions and the next timestep.
    """
    robotnames = []
    var_dict = {}
    for robot in robots:
        robotnames.append(robot.name)
        var_dict.update({Var("x_"+str(robot.name)+"_0"): robot.pos.x})
        var_dict.update({Var("y_"+str(robot.name)+"_0"): robot.pos.y})
        var_dict.update({Var("dist_"+str(robot.name)+"_0"): distance([[robot.pos.x, robot.pos.y]], [robot.goal])})

    robotcombis = combinations(robots, 2)
    cur_dist = np.inf
    for combi in list(robotcombis):
        distance_btw_robots = np.abs(combi[0].pos.x - combi[1].pos.x) + np.abs(combi[0].pos.y - combi[1].pos.y)
        cur_dist = min(cur_dist, distance_btw_robots)
    T_1 = T_0 + 1  # noqa: N806
    var_dict.update({Var("t_0"): T_0})
    var_dict.update({Var("t_1"): T_1})
    var_dict.update({Var("current_distance"): cur_dist})

    # current distance to the goal
    to_goal = distance([[robot.pos.x, robot.pos.y] for robot in robots], [robot.goal for robot in robots])
    var_dict.update({Var("cur_dist_to_goal"): to_goal})

    # next possible positions
    x_list = list({max(robots[0].pos.x - 1, 0), robots[0].pos.x, min(robots[0].pos.x + 1, grid_n)})
    y_list = list({max(robots[0].pos.y - 1, 0), robots[0].pos.y, min(robots[0].pos.y + 1, grid_m)})
    next_possible_positions = []
    for x in x_list:
        if x == robots[0].pos.x:
            for y in y_list:
                next_possible_positions.append([(x,y)])
        else:
            y = robots[0].pos.y
            next_possible_positions.append([(x,y)])

    for robot in robots[1:]:
        x_list = list({max(robot.pos.x - 1, 0), robot.pos.x, min(robot.pos.x + 1, grid_n)})
        y_list = list({max(robot.pos.y - 1, 0), robot.pos.y, min(robot.pos.y + 1, grid_m)})
        pos = []
        for x in x_list:
            if x == robot.pos.x:
                for y in y_list:
                    pos.append([(x,y)])
            else:
                y = robot.pos.y
                pos.append([(x,y)])

        # removing collision states
        next_list = []
        for pos_position in next_possible_positions:
            for new_robot_pos in pos:
                if new_robot_pos[0] not in pos_position:
                    entry = pos_position + new_robot_pos
                    next_list.append(entry)

        next_possible_positions = next_list

    # remove a random element from list
    while len(next_possible_positions) > 0:
        possible_position = random.choice(next_possible_positions)
        next_possible_positions.remove(possible_position)

        pos_dict = {}
        for i,robot in enumerate(robots):
            pos_dict.update({robot.name: possible_position[i]})
            var_dict.update({Var("x_"+str(robot.name)+"_1"): possible_position[i][0]})
            var_dict.update({Var("y_"+str(robot.name)+"_1"): possible_position[i][1]})
            var_dict.update({Var("dist_"+str(robot.name)+"_1"): distance([[possible_position[i][0], possible_position[i][1]]], [robot.goal])})

        # next distance to the goal
        next_to_goal = distance([[possible_position[i][0], possible_position[i][1]] for i in range(len(possible_position))], [robot.goal for robot in robots])
        var_dict.update({Var("next_dist_to_goal"): next_to_goal})

        if contract.a.contains_behavior(var_dict) and contract.g.contains_behavior(var_dict):
            passed = True
            for c in contractlist_1:
                if not c.a.contains_behavior(var_dict) or not c.g.contains_behavior(var_dict):
                    passed = False
                    break;
            for c in contractlist_2:
                if not c.a.contains_behavior(var_dict) or not c.g.contains_behavior(var_dict):
                    passed = False
                    break;
            if passed:
                return possible_position, T_1
    return [], T_1

def check_move(var_dict, collision_contract_list, swapping_contract_list):
    passed = True
    for c in swapping_contract_list:
        if not c.g.contains_behavior(var_dict):
            passed = False
            break;

    for c in collision_contract_list:
        if not c.a.contains_behavior(var_dict) or not c.g.contains_behavior(var_dict):
            passed = False
            break;
    if passed:
        return True
    return False


def find_random_move(  # noqa: WPS231
    grid_n: int, grid_m: int, robots: List[Robot], T_0: int, contract: PolyhedralIoContractCompound, # noqa: N803
    contractlist_1 : List[PolyhedralIoContractCompound], contractlist_2 : List[PolyhedralIoContractCompound] ) -> tuple[list, int]:
    """
    Evaluate the contracts for possible next positions of the robots to find allowed moves.

    Args:
        grid_n: Grid dimension n.
        grid_m: Grid dimension m.
        robots: list of robots.
        T_0: Current time step.
        contract: The contract that the next positions must satisfy.
        contractlist_1: The list of compound contracts that the next positions must satisfy.
        contractlist_2: Another list of compound contracts that the next positions must satisfy.

    Returns:
        A random move containing the next positions and the next timestep, that satisfies all contracts.
    """
    robotnames = []
    var_dict = {}
    for robot in robots:
        robotnames.append(robot.name)
        var_dict.update({Var("x_"+str(robot.name)+"_0"): robot.pos.x})
        var_dict.update({Var("y_"+str(robot.name)+"_0"): robot.pos.y})
        var_dict.update({Var("dist_"+str(robot.name)+"_0"): distance([[robot.pos.x, robot.pos.y]], [robot.goal])})

    robotcombis = combinations(robots, 2)
    cur_dist = np.inf
    for combi in list(robotcombis):
        distance_btw_robots = np.abs(combi[0].pos.x - combi[1].pos.x) + np.abs(combi[0].pos.y - combi[1].pos.y)
        cur_dist = min(cur_dist, distance_btw_robots)
    T_1 = T_0 + 1  # noqa: N806
    var_dict.update({Var("t_0"): T_0})
    var_dict.update({Var("t_1"): T_1})
    var_dict.update({Var("current_distance"): cur_dist})

    # current distance to the goal
    to_goal = distance([[robot.pos.x, robot.pos.y] for robot in robots], [robot.goal for robot in robots])
    var_dict.update({Var("cur_dist_to_goal"): to_goal})

    # next possible positions
    x_list = list({max(robots[0].pos.x - 1, 0), robots[0].pos.x, min(robots[0].pos.x + 1, grid_n)})
    y_list = list({max(robots[0].pos.y - 1, 0), robots[0].pos.y, min(robots[0].pos.y + 1, grid_m)})
    next_possible_positions = []

    for x in x_list:
        if x == robots[0].pos.x:
            for y in y_list:
                next_possible_positions.append([(x,y)])
        else:
            y = robots[0].pos.y
            next_possible_positions.append([(x,y)])

    for robot in robots[1:]:
        x_list = list({max(robot.pos.x - 1, 0), robot.pos.x, min(robot.pos.x + 1, grid_n)})
        y_list = list({max(robot.pos.y - 1, 0), robot.pos.y, min(robot.pos.y + 1, grid_m)})
        pos = []
        for x in x_list:
            if x == robot.pos.x:
                for y in y_list:
                    pos.append([(x,y)])
            else:
                y = robot.pos.y
                pos.append([(x,y)])

        # removing collision states
        next_list = []
        for pos_position in next_possible_positions:
            for new_robot_pos in pos:
                if new_robot_pos[0] not in pos_position:
                    entry = pos_position + new_robot_pos
                    next_list.append(entry)

        next_possible_positions = next_list

    # pick a random element from list and remove
    while len(next_possible_positions) > 0:
        possible_position = random.choice(next_possible_positions)
        next_possible_positions.remove(possible_position)

        pos_dict = {}
        for i,robot in enumerate(robots):
            pos_dict.update({robot.name: possible_position[i]})
            var_dict.update({Var("x_"+str(robot.name)+"_1"): possible_position[i][0]})
            var_dict.update({Var("y_"+str(robot.name)+"_1"): possible_position[i][1]})
            var_dict.update({Var("dist_"+str(robot.name)+"_1"): distance([[possible_position[i][0], possible_position[i][1]]], [robot.goal])})

        # next distance to the goal
        next_to_goal = distance([[possible_position[i][0], possible_position[i][1]] for i in range(len(possible_position))], [robot.goal for robot in robots])
        var_dict.update({Var("next_dist_to_goal"): next_to_goal})

        if contract.a.contains_behavior(var_dict) and contract.g.contains_behavior(var_dict):
            passed = True
            for c in contractlist_1:
                if not c.a.contains_behavior(var_dict) or not c.g.contains_behavior(var_dict):
                    passed = False
                    break;
            for c in contractlist_2:
                if not c.g.contains_behavior(var_dict):
                    passed = False
                    break;
            if passed:
                return possible_position, T_1
    return [], T_1
