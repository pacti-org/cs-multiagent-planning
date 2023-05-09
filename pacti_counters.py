from functools import wraps
from typing import List, Optional
from pacti.terms.polyhedra import PolyhedralContract

import sys
import functools
import dataclasses


@functools.total_ordering
@dataclasses.dataclass
class PolyhedralContractSize:
    # The count of assumptions and guarantees.
    constraints: int = 0

    # The count of input and output variables.
    variables: int = 0

    def __init__(self, contract: Optional[PolyhedralContract], max_values: bool = False):
        if max_values:
            self.constraints = sys.maxsize
            self.variables = sys.maxsize
        elif contract:
            self.constraints = len(contract.a.terms) + len(contract.g.terms)
            self.variables = len(contract.vars)
        else:
            self.constraints = 0
            self.variables = 0

    def __eq__(self, other: "PolyhedralContractSize") -> bool:
        return (self.constraints == other.constraints) and (self.variables == other.variables)

    def __ge__(self, other: "PolyhedralContractSize") -> bool:
        return (self.constraints, self.variables) >= (other.constraints, other.variables)

    def max(self, other: "PolyhedralContractSize") -> "PolyhedralContractSize":
        if self >= other:
            return self
        else:
            return other

    @staticmethod
    def _render_(value: int) -> str:
        if sys.maxsize == value:
            return "inf"
        return str(value)

    def __str__(self) -> str:
        return f"(constraints: {PolyhedralContractSize._render_(self.constraints)}, variables: {PolyhedralContractSize._render_(self.variables)})"


def statistics_decorator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        wrapper.counter += 1

        # Call the original method to get the resulting contract
        result_contract = fn(*args, **kwargs)

        # Calculate the size of the input contracts and the result contract
        input_size1 = PolyhedralContractSize(contract=args[0])
        input_size2 = PolyhedralContractSize(contract=args[1])
        result_size = PolyhedralContractSize(contract=result_contract)

        # Update the min/max polyhedral contract size if necessary
        wrapper.min_size = min(wrapper.min_size, input_size1, input_size2, result_size)
        wrapper.max_size = max(wrapper.max_size, input_size1, input_size2, result_size)

        # Return the composed contract
        return result_contract

    wrapper.counter = 0
    wrapper.min_size = PolyhedralContractSize(contract=None, max_values=True)
    wrapper.max_size = PolyhedralContractSize(contract=None, max_values=False)
    return wrapper


PolyhedralContract.compose = statistics_decorator(PolyhedralContract.compose)
PolyhedralContract.quotient = statistics_decorator(PolyhedralContract.quotient)
PolyhedralContract.merge = statistics_decorator(PolyhedralContract.merge)


@dataclasses.dataclass
class PolyhedralContractCounts:
    compose_count: int = 0
    quotient_count: int = 0
    merge_count: int = 0

    compose_min_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=True)
    )
    quotient_min_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=True)
    )
    merge_min_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=True)
    )

    compose_max_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=False)
    )
    quotient_max_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=False)
    )
    merge_max_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=False)
    )

    def reset(self) -> None:
        self.compose_count = 0
        self.quotient_count = 0
        self.merge_count = 0
        self.compose_min_size = PolyhedralContractSize(contract=None, max_values=True)
        self.quotient_min_size = PolyhedralContractSize(contract=None, max_values=True)
        self.merge_min_size = PolyhedralContractSize(contract=None, max_values=True)
        self.compose_max_size = PolyhedralContractSize(contract=None, max_values=False)
        self.quotient_max_size = PolyhedralContractSize(contract=None, max_values=False)
        self.merge_max_size = PolyhedralContractSize(contract=None, max_values=False)

    def update_counts(self) -> "PolyhedralContractCounts":
        self.compose_count = PolyhedralContract.compose.counter
        self.quotient_count = PolyhedralContract.quotient.counter
        self.merge_count = PolyhedralContract.merge.counter

        self.compose_min_size = PolyhedralContract.compose.min_size
        self.quotient_min_size = PolyhedralContract.quotient.min_size
        self.merge_min_size = PolyhedralContract.merge.min_size

        self.compose_max_size = PolyhedralContract.compose.max_size
        self.quotient_max_size = PolyhedralContract.quotient.max_size
        self.merge_max_size = PolyhedralContract.merge.max_size

        return self

    def __add__(self, other: "PolyhedralContractCounts") -> "PolyhedralContractCounts":
        result = PolyhedralContractCounts()
        result.compose_count = self.compose_count + other.compose_count
        result.quotient_count = self.quotient_count + other.quotient_count
        result.merge_count = self.merge_count + other.merge_count

        result.compose_min_size = min(self.compose_min_size, other.compose_min_size)
        result.quotient_min_size = min(self.quotient_min_size, other.quotient_min_size)
        result.merge_min_size = min(self.merge_min_size, other.merge_min_size)

        result.compose_max_size = max(self.compose_max_size, other.compose_max_size)
        result.quotient_max_size = max(self.quotient_max_size, other.quotient_max_size)
        result.merge_max_size = max(self.merge_max_size, other.merge_max_size)

        return result

    def __str__(self) -> str:
        def operation_msg(operation_count: int, min_size: PolyhedralContractSize, max_size: PolyhedralContractSize, operation_name: str) -> str:
            if operation_count == 0:
                return f"no {operation_name} operations\n"
            else:
                return f"min/max sizes for {operation_name}={min_size}/{max_size}\n"

        compose_msg = operation_msg(self.compose_count, self.compose_min_size, self.compose_max_size, "compose")
        quotient_msg = operation_msg(self.quotient_count, self.quotient_min_size, self.quotient_max_size, "quotient")
        merge_msg = operation_msg(self.merge_count, self.merge_min_size, self.merge_max_size, "merge")

        return (
            f"PolyhedralContract operation counts: compose={self.compose_count}, quotient={self.quotient_count}, merge={self.merge_count}.\n"
            + compose_msg
            + quotient_msg
            + merge_msg
        )


@dataclasses.dataclass
class PolyhedralContractCountStats:
    min_compose: int = 0
    min_quotient: int = 0
    min_merge: int = 0

    max_compose: int = 0
    max_quotient: int = 0
    max_merge: int = 0

    avg_compose: float = 0
    avg_quotient: float = 0
    avg_merge: float = 0

    compose_min_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=True)
    )
    quotient_min_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=True)
    )
    merge_min_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=True)
    )

    compose_max_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=False)
    )
    quotient_max_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=False)
    )
    merge_max_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=False)
    )

    def stats(self) -> str:
        def operation_stats(operation_name: str, min_count: int, max_count: int, avg_count: float, min_size: PolyhedralContractSize, max_size: PolyhedralContractSize) -> str:
            if max_count == 0:
                return f"no {operation_name} operations\n"
            else:
                return (
                    f"{operation_name} invocation counts: (min: {min_count}, max: {max_count}, avg: {avg_count})\n"
                    + f"min/max {operation_name} contract size: {min_size}/{max_size}\n"
                )

        compose_stats = operation_stats("compose", self.min_compose, self.max_compose, self.avg_compose, self.compose_min_size, self.compose_max_size)
        quotient_stats = operation_stats("quotient", self.min_quotient, self.max_quotient, self.avg_quotient, self.quotient_min_size, self.quotient_max_size)
        merge_stats = operation_stats("merge", self.min_merge, self.max_merge, self.avg_merge, self.merge_min_size, self.merge_max_size)

        return (
            "Pacti compose,quotient,merge statistics:\n"
            + compose_stats
            + quotient_stats
            + merge_stats
        )

def polyhedral_count_stats(counts: List[PolyhedralContractCounts]) -> PolyhedralContractCountStats:
    stats = PolyhedralContractCountStats()
    # Minimum counts
    stats.min_compose = min(count.compose_count for count in counts)
    stats.min_quotient = min(count.quotient_count for count in counts)
    stats.min_merge = min(count.merge_count for count in counts)

    # Maximum counts
    stats.max_compose = max(count.compose_count for count in counts)
    stats.max_quotient = max(count.quotient_count for count in counts)
    stats.max_merge = max(count.merge_count for count in counts)

    # Average counts
    n = len(counts)
    stats.avg_compose: float = sum(count.compose_count for count in counts) / n
    stats.avg_quotient: float = sum(count.quotient_count for count in counts) / n
    stats.avg_merge: float = sum(count.merge_count for count in counts) / n

    # Minimum PolyhedralContractSize for each operation
    stats.compose_min_size = min(count.compose_min_size for count in counts)
    stats.quotient_min_size = min(count.quotient_min_size for count in counts)
    stats.merge_min_size = min(count.merge_min_size for count in counts)

    # Maximum PolyhedralContractSize for each operation
    stats.compose_max_size = max(count.compose_max_size for count in counts)
    stats.quotient_max_size = max(count.quotient_max_size for count in counts)
    stats.merge_max_size = max(count.merge_max_size for count in counts)

    return stats
