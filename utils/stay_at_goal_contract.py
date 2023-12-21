from pacti.contracts import PolyhedralIoContract
from typing import List

from utils.multiagent_utils import (
    Coord,
    Robot,
    distance
)

def robots_stay_at_goal(robots: List[Robot], timestep: int):
    contracts = []
    for robot in robots:
        contract = stay_at_goal_coord(robot, timestep)
        contracts.append(contract)

    goal_contract = contracts[0]
    for contract in contracts[1:]:
        goal_contract= goal_contract.merge(contract)
    return goal_contract

def stay_at_goal_coord(robot: Robot, timestep: int):
    cur_dist_val = distance([[robot.pos.x, robot.pos.y]], [robot.goal])

    x_str_0 = 'x_{0}_0'.format(robot.name)
    y_str_0 = 'y_{0}_0'.format(robot.name)
    x_str_1 = 'x_{0}_1'.format(robot.name)
    y_str_1 = 'y_{0}_1'.format(robot.name)
    t_0 = 't_0'
    t_1 = 't_1'

    asm = ["{0} = {1}".format(t_0, timestep)]
    guar = ["{0} - {1} = 1".format(t_1, t_0)]

    if cur_dist_val == 0:
        asm.append("{0} = {1}".format(x_str_0, robot.pos.x))
        asm.append("{0} = {1}".format(y_str_0, robot.pos.y))
        guar.append("{0} = {1}".format(x_str_1, robot.pos.x))
        guar.append("{0} = {1}".format(y_str_1, robot.pos.y))

    contract = PolyhedralIoContract.from_strings(
            input_vars=[x_str_0, y_str_0, t_0],
            output_vars=[x_str_1, y_str_1, t_1],
            assumptions=asm,
            guarantees=guar,
        )

    return contract
