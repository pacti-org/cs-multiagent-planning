from pacti.contracts import PolyhedralIoContractCompound, PolyhedralIoContract
from utils.multiagent_utils import Coord
from typing import List


def get_dynamics_contract(name: str, init_pos: Coord, timestep: int, grid_n: int, grid_m: int
) -> PolyhedralIoContractCompound:  # noqa: WPS210
    """
    Function to set up the contract encoding the dynamics for a single robot for the next timestep.

    Args:
        name: name of the robot
        init_pos: current coordinates of the robot
        timestep: current timestep

    Returns:
        Resulting contract compound that encodes the dynamics for the robot.
    """
    x_str_0 = 'x_{0}_0'.format(name)
    y_str_0 = 'y_{0}_0'.format(name)
    x_str_1 = 'x_{0}_1'.format(name)
    y_str_1 = 'y_{0}_1'.format(name)
    t_0 = 't_0'
    t_1 = 't_1'

    contract = PolyhedralIoContract.from_strings(
        input_vars=[x_str_0, y_str_0, t_0],
        output_vars=[x_str_1, y_str_1, t_1],
        assumptions=[
                "{0} = {1}".format(x_str_0, init_pos.x),
                "{0} = {1}".format(y_str_0, init_pos.y),
                "{0} = {1}".format(t_0, timestep),
        ],
        guarantees=[
                "{0} - {1} = 1".format(t_1, t_0),
                "{0} - {1} + {2} - {3} <= 1".format(x_str_1, x_str_0, y_str_1, y_str_0),
                "{0} - {1} - {2} + {3} <= 1".format(x_str_1, x_str_0, y_str_1, y_str_0),
                "-{0} + {1} + {2} - {3} <= 1".format(x_str_1, x_str_0, y_str_1, y_str_0),
                "-{0} + {1} - {2} + {3} <= 1".format(x_str_1, x_str_0, y_str_1, y_str_0),
                "{0} <= {1}".format(x_str_1, grid_n - 1),
                "{0} <= {1}".format(y_str_1, grid_m - 1),
                "-{0} <= 0".format(x_str_1),
                "-{0} <= 0".format(y_str_1),
        ],
    )
    return contract

def addl_dynamics_requirement_robot(robot: str, addl_guarantees: List[str]
) -> PolyhedralIoContractCompound:  # noqa: WPS210
    """
    Function to set up the contract encoding the dynamics for a single robot for the next timestep.

    Args:
        name: name of the robot
        init_pos: current coordinates of the robot
        timestep: current timestep

    Returns:
        Resulting contract compound that encodes the dynamics for the robot.
    """
    x_str_0 = 'x_{0}_0'.format(robot.name)
    y_str_0 = 'y_{0}_0'.format(robot.name)
    x_str_1 = 'x_{0}_1'.format(robot.name)
    y_str_1 = 'y_{0}_1'.format(robot.name)


    contract = PolyhedralIoContract.from_strings(
        input_vars=[x_str_0, y_str_0],
        output_vars=[x_str_1, y_str_1],
        assumptions=[
                "{0} = {1}".format(x_str_0, robot.pos.x),
                "{0} = {1}".format(y_str_0, robot.pos.y),
        ],
        guarantees= addl_guarantees,
    )
    return contract
