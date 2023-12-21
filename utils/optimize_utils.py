

from pacti.terms.polyhedra import PolyhedralTermList, PolyhedralTerm
from scipy.optimize import linprog
from pacti.terms.polyhedra import serializer
from pacti.contracts import PolyhedralIoContract

def optimize_tlist(term_list, objective: dict, maximize: bool = True):
    """
    Optimizes a linear expression in the feasible region of the termlist.

    Args:
        objective:
            The objective to optimize.
        maximize:
            If true, the routine maximizes; it minimizes otherwise.

    Returns:
        The optimal value of the objective. If the objective is unbounded, None is returned.

    Raises:
        ValueError: Constraints are likely unfeasible.
    """
    obj = PolyhedralTermList([PolyhedralTerm(variables=objective, constant=0)])
    variables, self_mat, self_cons, obj_mat, _ = PolyhedralTermList.termlist_to_polytope(term_list, obj)  # noqa: WPS236
    polarity = 1
    if maximize:
        polarity = -1
    res = linprog(c=polarity * obj_mat[0], A_ub=self_mat, b_ub=self_cons, bounds=(None, None))
    # Linprog's status values
    # 0 : Optimization proceeding nominally.
    # 1 : Iteration limit reached.
    # 2 : Problem appears to be infeasible.
    # 3 : Problem appears to be unbounded.
    # 4 : Numerical difficulties encountered.
    if res["status"] == 3:
        return None
    elif res["status"] == 0:
        fun_val: float = res["fun"]
        return polarity * fun_val, {variables[i]:res["x"][i] for i in range(len(variables))}
    raise ValueError("Constraints are unfeasible")


def optimize_contract(contract, expr: str, maximize: bool = True):
    """Optimize linear objective over the contract.

    Compute the optima of a linear objective over the assumptions and
    guarantees of the contract.

    Args:
        expr:
            linear objective being optimized.
        maximize:
            Maximize if True; minimize if False.

    Returns:
        The optimal value of the objective in the context of the contract.
    """
    new_expr = expr + " <= 0"
    variables = serializer.polyhedral_termlist_from_string(new_expr)[0].variables
    constant = serializer.polyhedral_termlist_from_string(new_expr)[0].constant
    constraints: PolyhedralTermList = contract.a | contract.g
    obj, vars = optimize_tlist(constraints, variables, maximize)
    return obj - constant, vars


if __name__=="__main__":
    cont = PolyhedralIoContract.from_strings(
        input_vars=["x"],
        output_vars=["y"],
        assumptions=[],
        guarantees=["y <= 5", "x <= 2y"]
        )

    print(optimize_contract(contract=cont, expr="y", maximize=True))
