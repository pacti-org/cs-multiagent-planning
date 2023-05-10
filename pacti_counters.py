from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pacti.terms.polyhedra import PolyhedralContract, PolyhedralTermList, PolyhedralContractCompound
from pacti.utils.lists import list_union

import sys
import functools
import dataclasses


@functools.total_ordering
@dataclasses.dataclass
class PolyhedralContractSize:
    """
    A data class representing the size of a PolyhedralContract.
    The size is determined by the number of constraints and variables.
    """

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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolyhedralContractSize):
            return NotImplemented
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
        return (
            f"(constraints: {PolyhedralContractSize._render_(self.constraints)},"
            f" variables: {PolyhedralContractSize._render_(self.variables)})"
        )


class ContractWrapper:
    def __init__(self, fn: Callable[..., PolyhedralContract]):
        self.fn = fn
        self.counter: int = 0
        self.min_size: PolyhedralContractSize = PolyhedralContractSize(contract=None, max_values=True)
        self.max_size: PolyhedralContractSize = PolyhedralContractSize(contract=None, max_values=False)

    def __call__(
        self, instance: PolyhedralContract, *args: Tuple[PolyhedralContract], **kwargs: Dict[str, Any]
    ) -> PolyhedralContract:
        self.counter += 1

        # Call the original method to get the resulting contract
        result_contract = self.fn(instance, *args, **kwargs)

        # Calculate the size of the input contracts and the result contract
        input_size1 = PolyhedralContractSize(contract=instance)
        input_size2 = PolyhedralContractSize(contract=args[0])
        result_size = PolyhedralContractSize(contract=result_contract)

        # Update the min/max polyhedral contract size if necessary
        self.min_size = min(self.min_size, input_size1, input_size2, result_size)
        self.max_size = max(self.max_size, input_size1, input_size2, result_size)

        # Return the composed contract
        return result_contract


def contract_statistics_decorator(
    fn: Callable[..., PolyhedralContract]
) -> Callable[..., PolyhedralContract]:
    """
    A decorator to gather statistics about PolyhedralContract composition operations.
    It keeps track of the number of times the decorated function is called,
    and updates the minimum and maximum sizes of input and output contracts.
    """

    wrapper = ContractWrapper(fn)

    def wrapped_fn(
        instance: PolyhedralContract,
        *args: Tuple[PolyhedralContract],
        **kwargs: Dict[str, Any],
    ) -> PolyhedralContract:
        return wrapper(instance, *args, **kwargs)

    # Attach the 'wrapper' attribute to the 'wrapped_fn'
    setattr(wrapped_fn, "wrapper", wrapper)
    return wrapped_fn


PolyhedralContract.compose = contract_statistics_decorator(PolyhedralContract.compose)
PolyhedralContract.quotient = contract_statistics_decorator(PolyhedralContract.quotient)
PolyhedralContract.merge = contract_statistics_decorator(PolyhedralContract.merge)


@functools.total_ordering
@dataclasses.dataclass
class PolyhedralTermListSize:
    """
    A data class representing the size of a PolyhedralTermList.
    The size is determined by the number of terms and variables.
    """

    # The count of terms.
    terms: int = 0

    # The count of input and output variables.
    variables: int = 0

    def __init__(self, termlist: Optional[PolyhedralTermList], max_values: bool = False):
        if max_values:
            self.terms = sys.maxsize
            self.variables = sys.maxsize
        elif termlist:
            self.terms = len(termlist.terms)
            self.variables = len(termlist.vars)
        else:
            self.terms = 0
            self.variables = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolyhedralTermListSize):
            return NotImplemented
        return (self.terms == other.terms) and (self.variables == other.variables)

    def __ge__(self, other: "PolyhedralTermListSize") -> bool:
        return (self.terms, self.variables) >= (other.terms, other.variables)

    def max(self, other: "PolyhedralTermListSize") -> "PolyhedralTermListSize":
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
        return (
            f"(terms: {PolyhedralTermListSize._render_(self.terms)},"
            f" variables: {PolyhedralTermListSize._render_(self.variables)})"
        )


class TermListWrapper:
    def __init__(self, fn: Callable[..., bool]):
        self.fn = fn
        self.counter: int = 0
        self.min_size: PolyhedralTermListSize = PolyhedralTermListSize(termlist=None, max_values=True)
        self.max_size: PolyhedralTermListSize = PolyhedralTermListSize(termlist=None, max_values=False)

    def __call__(self, instance: PolyhedralTermList, *args: Tuple, **kwargs: Dict[str, Any]) -> bool:
        self.counter += 1

        # Call the original method to get the resulting term list
        result = self.fn(instance, *args, **kwargs)

        # Calculate the size of the input termlist
        input_size = PolyhedralTermListSize(termlist=instance)

        # Update the min/max polyhedral termlist size if necessary
        self.min_size = min(self.min_size, input_size)
        self.max_size = max(self.max_size, input_size)

        # Return the result
        return result


def termlist_statistics_decorator(fn: Callable[..., bool]) -> Callable[..., bool]:
    """
    A decorator to gather statistics about PolyhedralTermList operations.
    It keeps track of the number of times the decorated function is called,
    and updates the minimum and maximum sizes of input term lists.
    """

    wrapper = TermListWrapper(fn)

    def wrapped_fn(
        instance: PolyhedralTermList,
        *args: Tuple,
        **kwargs: Dict[str, Any],
    ) -> bool:
        return wrapper(instance, *args, **kwargs)

    # Attach the 'wrapper' attribute to the 'wrapped_fn'
    setattr(wrapped_fn, "wrapper", wrapper)
    return wrapped_fn


PolyhedralTermList.contains_behavior = termlist_statistics_decorator(PolyhedralTermList.contains_behavior)


@functools.total_ordering
@dataclasses.dataclass
class PolyhedralContractCompoundSize:
    """
    A data class representing the size of a PolyhedralContract.
    The size is determined by the number of constraints and variables.
    """

    # The count of convex polyhedra (assumptions and guarantees).
    convex_polyhedra_count: int = 0

    # The count of input and output variables.
    variables: int = 0

    def __init__(self, contract: Optional[PolyhedralContractCompound], max_values: bool = False):
        if max_values:
            self.convex_polyhedra_count = sys.maxsize
            self.variables = sys.maxsize
        elif contract:
            self.convex_polyhedra_count = len(contract.a.nested_termlist) + len(contract.g.nested_termlist)
            self.variables = len(list_union(contract.a.vars, contract.g.vars))
        else:
            self.convex_polyhedra_count = 0
            self.variables = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolyhedralContractCompoundSize):
            return NotImplemented
        return (self.convex_polyhedra_count == other.convex_polyhedra_count) and (self.variables == other.variables)

    def __ge__(self, other: "PolyhedralContractCompoundSize") -> bool:
        return (self.convex_polyhedra_count, self.variables) >= (other.convex_polyhedra_count, other.variables)

    def max(self, other: "PolyhedralContractCompoundSize") -> "PolyhedralContractCompoundSize":
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
        return (
            f"(constraints: {PolyhedralContractCompoundSize._render_(self.convex_polyhedra_count)},"
            f" variables: {PolyhedralContractCompoundSize._render_(self.variables)})"
        )


class CompoundContractWrapper:
    def __init__(self, fn: Callable[..., PolyhedralContractCompound]):
        self.fn = fn
        self.counter: int = 0
        self.min_size: PolyhedralContractCompoundSize = PolyhedralContractCompoundSize(contract=None, max_values=True)
        self.max_size: PolyhedralContractCompoundSize = PolyhedralContractCompoundSize(contract=None, max_values=False)

    def __call__(
        self, instance: PolyhedralContractCompound, *args: Tuple[PolyhedralContractCompound], **kwargs: Dict[str, Any]
    ) -> PolyhedralContractCompound:
        self.counter += 1

        # Call the original method to get the resulting contract
        result_contract = self.fn(instance, *args, **kwargs)

        # Calculate the size of the input contracts and the result contract
        input_size1 = PolyhedralContractCompoundSize(contract=instance)
        input_size2 = PolyhedralContractCompoundSize(contract=args[0])
        result_size = PolyhedralContractCompoundSize(contract=result_contract)

        # Update the min/max polyhedral contract size if necessary
        self.min_size = min(self.min_size, input_size1, input_size2, result_size)
        self.max_size = max(self.max_size, input_size1, input_size2, result_size)

        # Return the composed contract
        return result_contract


def compound_contract_statistics_decorator(
    fn: Callable[..., PolyhedralContractCompound]
) -> Callable[..., PolyhedralContractCompound]:
    """
    A decorator to gather statistics about PolyhedralTermList operations.
    It keeps track of the number of times the decorated function is called,
    and updates the minimum and maximum sizes of input term lists.
    """

    wrapper = CompoundContractWrapper(fn)

    def wrapped_fn(
        instance: PolyhedralContractCompound,
        *args: Tuple[PolyhedralContractCompound],
        **kwargs: Dict[str, Any],
    ) -> PolyhedralContractCompound:
        return wrapper(instance, *args, **kwargs)

    # Attach the 'wrapper' attribute to the 'wrapped_fn'
    setattr(wrapped_fn, "wrapper", wrapper)
    return wrapped_fn


PolyhedralContractCompound.merge = compound_contract_statistics_decorator(PolyhedralContractCompound.merge)


@dataclasses.dataclass
class PactiInstrumentationData:
    compose_count: int = 0
    quotient_count: int = 0
    merge_count: int = 0
    contains_behavior_count: int = 0
    compound_merge_count: int = 0

    compose_min_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=True)
    )
    quotient_min_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=True)
    )
    merge_min_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=True)
    )
    contains_behavior_min_size: PolyhedralTermListSize = dataclasses.field(
        default_factory=lambda: PolyhedralTermListSize(termlist=None, max_values=True)
    )
    compound_merge_min_size: PolyhedralContractCompoundSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractCompoundSize(contract=None, max_values=True)
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
    contains_behavior_max_size: PolyhedralTermListSize = dataclasses.field(
        default_factory=lambda: PolyhedralTermListSize(termlist=None, max_values=False)
    )
    compound_merge_max_size: PolyhedralContractCompoundSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractCompoundSize(contract=None, max_values=False)
    )

    def reset(self) -> None:
        self.compose_count = 0
        self.quotient_count = 0
        self.merge_count = 0
        self.contains_behavior_count = 0
        self.compound_merge_count = 0
        self.compose_min_size = PolyhedralContractSize(contract=None, max_values=True)
        self.quotient_min_size = PolyhedralContractSize(contract=None, max_values=True)
        self.merge_min_size = PolyhedralContractSize(contract=None, max_values=True)
        self.compose_max_size = PolyhedralContractSize(contract=None, max_values=False)
        self.quotient_max_size = PolyhedralContractSize(contract=None, max_values=False)
        self.merge_max_size = PolyhedralContractSize(contract=None, max_values=False)
        self.contains_behavior_min_size = PolyhedralTermListSize(termlist=None, max_values=True)
        self.contains_behavior_max_size = PolyhedralTermListSize(termlist=None, max_values=False)
        self.compound_merge_min_size = PolyhedralContractCompoundSize(contract=None, max_values=True)
        self.compound_merge_max_size = PolyhedralContractCompoundSize(contract=None, max_values=False)

    def update_counts(self) -> "PactiInstrumentationData":
        self.compose_count = PolyhedralContract.compose.wrapper.counter
        self.quotient_count = PolyhedralContract.quotient.wrapper.counter
        self.merge_count = PolyhedralContract.merge.wrapper.counter
        self.contains_behavior_count = PolyhedralTermList.contains_behavior.wrapper.counter
        self.compound_merge_count = PolyhedralContractCompound.merge.wrapper.counter

        self.compose_min_size = PolyhedralContract.compose.wrapper.min_size
        self.quotient_min_size = PolyhedralContract.quotient.wrapper.min_size
        self.merge_min_size = PolyhedralContract.merge.wrapper.min_size
        self.contains_behavior_min_size = PolyhedralTermList.contains_behavior.wrapper.min_size
        self.compound_merge_min_size = PolyhedralContractCompound.merge.wrapper.min_size

        self.compose_max_size = PolyhedralContract.compose.wrapper.max_size
        self.quotient_max_size = PolyhedralContract.quotient.wrapper.max_size
        self.merge_max_size = PolyhedralContract.merge.wrapper.max_size
        self.contains_behavior_max_size = PolyhedralTermList.contains_behavior.wrapper.max_size
        self.compound_merge_max_size = PolyhedralContractCompound.merge.wrapper.max_size

        return self

    def __add__(self, other: "PactiInstrumentationData") -> "PactiInstrumentationData":
        result = PactiInstrumentationData()
        result.compose_count = self.compose_count + other.compose_count
        result.quotient_count = self.quotient_count + other.quotient_count
        result.merge_count = self.merge_count + other.merge_count
        result.contains_behavior_count = self.contains_behavior_count + other.contains_behavior_count
        result.compound_merge_count = self.compound_merge_count + other.compound_merge_count

        result.compose_min_size = min(self.compose_min_size, other.compose_min_size)
        result.quotient_min_size = min(self.quotient_min_size, other.quotient_min_size)
        result.merge_min_size = min(self.merge_min_size, other.merge_min_size)
        result.contains_behavior_min_size = min(self.contains_behavior_min_size, other.contains_behavior_min_size)
        result.compound_merge_min_size = min(self.compound_merge_min_size, other.compound_merge_min_size)

        result.compose_max_size = max(self.compose_max_size, other.compose_max_size)
        result.quotient_max_size = max(self.quotient_max_size, other.quotient_max_size)
        result.merge_max_size = max(self.merge_max_size, other.merge_max_size)
        result.contains_behavior_max_size = max(self.contains_behavior_max_size, other.contains_behavior_max_size)
        result.compound_merge_max_size = max(self.compound_merge_max_size, other.compound_merge_max_size)

        return result

    def __str__(self) -> str:
        def operation_msg(
            operation_count: int,
            min_size: Union[PolyhedralContractSize, PolyhedralTermListSize, PolyhedralContractCompoundSize],
            max_size: Union[PolyhedralContractSize, PolyhedralTermListSize, PolyhedralContractCompoundSize],
            operation_name: str,
        ) -> str:
            if operation_count == 0:
                return f"no {operation_name} operations\n"
            else:
                return f"min/max sizes for {operation_name}={min_size}/{max_size}\n"

        compose_msg = operation_msg(self.compose_count, self.compose_min_size, self.compose_max_size, "compose")
        quotient_msg = operation_msg(self.quotient_count, self.quotient_min_size, self.quotient_max_size, "quotient")
        merge_msg = operation_msg(self.merge_count, self.merge_min_size, self.merge_max_size, "merge")

        contains_behavior_msg = operation_msg(
            self.contains_behavior_count,
            self.contains_behavior_min_size,
            self.contains_behavior_max_size,
            "contains_behavior",
        )
        compound_merge_msg = operation_msg(
            self.compound_merge_count, self.compound_merge_min_size, self.compound_merge_max_size, "compound_merge"
        )

        return (
            f"PolyhedralContract operation counts:"
            f" compose={self.compose_count},"
            f" quotient={self.quotient_count},"
            f" merge={self.merge_count}.\n"
            + compose_msg
            + quotient_msg
            + merge_msg
            + f"PolyhedralTermList operation counts: contains_behavior={self.contains_behavior_count}.\n"
            + contains_behavior_msg
            + f"PolyhedralContractCompound operation counts: merge={self.compound_merge_count}.\n"
            + compound_merge_msg
        )


@dataclasses.dataclass
class PactiInstrumentationSummary:
    min_compose: int = 0
    min_quotient: int = 0
    min_merge: int = 0
    min_contains_behavior: int = 0
    min_compound_merge: int = 0

    max_compose: int = 0
    max_quotient: int = 0
    max_merge: int = 0
    max_contains_behavior: int = 0
    max_compound_merge: int = 0

    total_compose: int = 0
    total_quotient: int = 0
    total_merge: int = 0
    total_contains_behavior: int = 0
    total_compound_merge: int = 0

    avg_compose: float = 0
    avg_quotient: float = 0
    avg_merge: float = 0
    avg_contains_behavior: float = 0
    avg_compound_merge: float = 0

    compose_min_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=True)
    )
    quotient_min_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=True)
    )
    merge_min_size: PolyhedralContractSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractSize(contract=None, max_values=True)
    )
    contains_behavior_min_size: PolyhedralTermListSize = dataclasses.field(
        default_factory=lambda: PolyhedralTermListSize(termlist=None, max_values=True)
    )
    compound_merge_min_size: PolyhedralContractCompoundSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractCompoundSize(contract=None, max_values=True)
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
    contains_behavior_max_size: PolyhedralTermListSize = dataclasses.field(
        default_factory=lambda: PolyhedralTermListSize(termlist=None, max_values=False)
    )
    compound_merge_max_size: PolyhedralContractCompoundSize = dataclasses.field(
        default_factory=lambda: PolyhedralContractCompoundSize(contract=None, max_values=False)
    )

    def __add__(self, other: "PactiInstrumentationSummary") -> "PactiInstrumentationSummary":
        stats = PactiInstrumentationSummary()
        stats.min_compose = min(self.min_compose, other.min_compose)
        stats.min_quotient = min(self.min_quotient, other.min_quotient)
        stats.min_merge = min(self.min_merge, other.min_merge)
        stats.min_contains_behavior = min(self.min_contains_behavior, other.min_contains_behavior)
        stats.min_compound_merge = min(self.min_compound_merge, other.min_compound_merge)

        stats.max_compose = max(self.max_compose, other.max_compose)
        stats.max_quotient = max(self.max_quotient, other.max_quotient)
        stats.max_merge = max(self.max_merge, other.max_merge)
        stats.max_contains_behavior = max(self.max_contains_behavior, other.max_contains_behavior)
        stats.max_compound_merge = max(self.max_compound_merge, other.max_compound_merge)

        stats.total_compose = self.total_compose + other.total_compose
        stats.total_quotient = self.total_quotient + other.total_quotient
        stats.total_merge = self.total_merge + other.total_merge
        stats.total_contains_behavior = self.total_contains_behavior + other.total_contains_behavior
        stats.total_compound_merge = self.total_compound_merge + other.total_compound_merge

        stats.avg_compose = (self.avg_compose + other.avg_compose) / 2
        stats.avg_quotient = (self.avg_quotient + other.avg_quotient) / 2
        stats.avg_merge = (self.avg_merge + other.avg_merge) / 2
        stats.avg_contains_behavior = (self.avg_contains_behavior + other.avg_contains_behavior) / 2
        stats.avg_compound_merge = (self.avg_compound_merge + other.avg_compound_merge) / 2

        stats.compose_min_size = min(self.compose_min_size, other.compose_min_size)
        stats.quotient_min_size = min(self.quotient_min_size, other.quotient_min_size)
        stats.merge_min_size = min(self.merge_min_size, other.merge_min_size)
        stats.contains_behavior_min_size = min(self.contains_behavior_min_size, other.contains_behavior_min_size)
        stats.compound_merge_min_size = min(self.compound_merge_min_size, other.compound_merge_min_size)

        stats.compose_max_size = max(self.compose_max_size, other.compose_max_size)
        stats.quotient_max_size = max(self.quotient_max_size, other.quotient_max_size)
        stats.merge_max_size = max(self.merge_max_size, other.merge_max_size)
        stats.contains_behavior_max_size = max(self.contains_behavior_max_size, other.contains_behavior_max_size)
        stats.compound_merge_max_size = max(self.compound_merge_max_size, other.compound_merge_max_size)

        return stats

    def stats(self) -> str:
        def operation_stats(
            operation_name: str,
            min_count: int,
            max_count: int,
            avg_count: float,
            total_count: int,
            min_size: Union[PolyhedralContractSize, PolyhedralTermListSize, PolyhedralContractCompoundSize],
            max_size: Union[PolyhedralContractSize, PolyhedralTermListSize, PolyhedralContractCompoundSize],
        ) -> str:
            if total_count == 0:
                return f"no {operation_name} operations\n"
            else:
                return (
                    f"{operation_name} invocation counts:"
                    f" (min: {min_count}"
                    f", max: {max_count}"
                    f", avg: {avg_count}"
                    f", total: {total_count})\n" + f"min/max {operation_name} contract size: {min_size}/{max_size}\n"
                )

        compose_stats = operation_stats(
            "compose",
            self.min_compose,
            self.max_compose,
            self.avg_compose,
            self.total_compose,
            self.compose_min_size,
            self.compose_max_size,
        )
        quotient_stats = operation_stats(
            "quotient",
            self.min_quotient,
            self.max_quotient,
            self.avg_quotient,
            self.total_quotient,
            self.quotient_min_size,
            self.quotient_max_size,
        )
        merge_stats = operation_stats(
            "merge",
            self.min_merge,
            self.max_merge,
            self.avg_merge,
            self.total_merge,
            self.merge_min_size,
            self.merge_max_size,
        )
        contains_behavior_stats = operation_stats(
            "contains_behavior",
            self.min_contains_behavior,
            self.max_contains_behavior,
            self.avg_contains_behavior,
            self.total_contains_behavior,
            self.contains_behavior_min_size,
            self.contains_behavior_max_size,
        )
        compound_merge_stats = operation_stats(
            "compound_merge",
            self.min_compound_merge,
            self.max_compound_merge,
            self.avg_compound_merge,
            self.total_compound_merge,
            self.compound_merge_min_size,
            self.compound_merge_max_size,
        )

        return (
            "Pacti compose,quotient,merge statistics:\n"
            + compose_stats
            + quotient_stats
            + merge_stats
            + "Pacti PolyhedralTermList statistics:\n"
            + contains_behavior_stats
            + "Pacti PolyhedralCompoundContract statistics:\n"
            + compound_merge_stats
        )


def summarize_instrumentation_data(counts: List[PactiInstrumentationData]) -> PactiInstrumentationSummary:
    stats = PactiInstrumentationSummary()
    # Minimum counts
    stats.min_compose = min(count.compose_count for count in counts)
    stats.min_quotient = min(count.quotient_count for count in counts)
    stats.min_merge = min(count.merge_count for count in counts)
    stats.min_contains_behavior = min(count.contains_behavior_count for count in counts)
    stats.min_compound_merge = min(count.compound_merge_count for count in counts)

    # Maximum counts
    stats.max_compose = max(count.compose_count for count in counts)
    stats.max_quotient = max(count.quotient_count for count in counts)
    stats.max_merge = max(count.merge_count for count in counts)
    stats.max_contains_behavior = max(count.contains_behavior_count for count in counts)
    stats.max_compound_merge = max(count.compound_merge_count for count in counts)

    # Total counts
    stats.total_compose = sum(count.compose_count for count in counts)
    stats.total_quotient = sum(count.quotient_count for count in counts)
    stats.total_merge = sum(count.merge_count for count in counts)
    stats.total_contains_behavior = sum(count.contains_behavior_count for count in counts)
    stats.total_compound_merge = sum(count.compound_merge_count for count in counts)

    # Average counts
    n = len(counts)
    stats.avg_compose = stats.total_compose / n
    stats.avg_quotient = stats.total_quotient / n
    stats.avg_merge = stats.total_merge / n
    stats.avg_contains_behavior = stats.total_contains_behavior / n
    stats.avg_compound_merge = stats.total_compound_merge / n

    # Minimum PolyhedralContractSize for each operation
    stats.compose_min_size = min(count.compose_min_size for count in counts)
    stats.quotient_min_size = min(count.quotient_min_size for count in counts)
    stats.merge_min_size = min(count.merge_min_size for count in counts)
    stats.contains_behavior_min_size = min(count.contains_behavior_min_size for count in counts)
    stats.compound_merge_min_size = min(count.compound_merge_min_size for count in counts)

    # Maximum PolyhedralContractSize for each operation
    stats.compose_max_size = max(count.compose_max_size for count in counts)
    stats.quotient_max_size = max(count.quotient_max_size for count in counts)
    stats.merge_max_size = max(count.merge_max_size for count in counts)
    stats.contains_behavior_max_size = max(count.contains_behavior_max_size for count in counts)
    stats.compound_merge_max_size = max(count.compound_merge_max_size for count in counts)

    return stats
