from typing import Dict, Any, Tuple
from pacti.iocontract import Var
from pacti.contracts import PolyhedralIoContractCompound
from pacti.contracts.polyhedral_iocontract import NestedPolyhedra
from pacti.terms.polyhedra import PolyhedralTermList, PolyhedralTerm
from pacti.utils.lists import list_diff

from itertools import combinations
import numpy as np


def tl_evaluate(this_tl, var_values: Dict) -> Tuple[bool, Any]:  # noqa: WPS231
    new_list = []
    for term in this_tl.terms:
        new_term = term.copy()
        for var, val in var_values.items():  # noqa: VNE002
            new_term = new_term.substitute_variable(
                var=var, subst_with_term=PolyhedralTerm(variables={}, constant=-val)
            )
        # we may have eliminated all variables after substitution
        if not new_term.vars:
            if new_term.constant < 0:
                return False, term
            else:
                continue  # noqa: WPS503
        new_list.append(new_term)
    return True, None

def tl_contains_behavior(this_tl, behavior: Dict) -> bool:
    excess_vars = list_diff(this_tl.vars, list(behavior.keys()))
    if excess_vars:
        raise ValueError("The variables %s were not assigned values" % (excess_vars))
    return tl_evaluate(this_tl,behavior)



def ntl_contains_behavior(nested_term_list:NestedPolyhedra, behavior: Dict) -> bool:
    """
    Tell whether constraints contain the given behavior.

    Args:
        behavior:
            The behavior in question.

    Returns:
        True if the behavior satisfies the constraints; false otherwise.

    Raises:
        ValueError: Not all variables in the constraints were assigned values.
    """
    result = []
    for tl in nested_term_list.nested_termlist:
        status, term = tl_contains_behavior(tl,behavior)
        result.append((status, term))
    return result

def extend_var_dict(var_dict, robots):

    vars = {}
    for key in var_dict.keys():
        vars.update({key.name: var_dict[key]})
    # var_names = [("x_"+robot.name+"_1", "y_"+robot.name+"_1") for robot in robots]
    # move = [(vars[xvar], vars[yvar]) for (xvar,yvar) in var_names]

    robotcombis = combinations(robots, 2)
    cur_dist = np.inf
    for combi in list(robotcombis):
        distance_btw_robots = np.abs(combi[0].pos.x - combi[1].pos.x) + np.abs(combi[0].pos.y - combi[1].pos.y)
        cur_dist = min(cur_dist, distance_btw_robots)
        # del_x_str = "delta_x_" + str(combi[0].name) + "_" + str(str(combi[1].name))
        # del_y_str = "delta_y_" + str(combi[0].name) + "_" + str(str(combi[1].name))
        # del_x = (vars['x_'+ combi[0].name+'_1'] - vars['x_'+ combi[1].name+'_1'] ) * (combi[0].pos.x - combi[1].pos.x)
        # del_y = (vars['y_'+ combi[0].name+'_1'] - vars['y_'+ combi[1].name+'_1'] ) * (combi[0].pos.y - combi[1].pos.y)
        # var_dict.update({Var(del_x_str): del_x})
        # var_dict.update({Var(del_y_str): del_y})
    var_dict.update({Var("current_distance"): cur_dist})

    return var_dict


if __name__=="__main__":
    cont = PolyhedralIoContractCompound.from_strings(
        input_vars="x",
        output_vars="y",
        assumptions=[["x <= 2"],["x >= 10"]],
        guarantees=[["y <= 5 + x"]]
        )
    x = Var("x")
    y = Var("y")
    print(ntl_contains_behavior(cont.a,behavior={x:2,y:5}))
