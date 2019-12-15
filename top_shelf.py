import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, lru_cache
from typing import *

import hypothesis.strategies as st
from hypothesis import given, infer, settings

Scalar = Union[AnyStr, int, bool]

ScalarToScalar = Callable[[Scalar], Scalar]

IntToInt = Callable[[int], int]

ScalarOrScalarFunction = Union[Scalar, ScalarToScalar]

RegularFunction = Callable[[ScalarOrScalarFunction], ScalarOrScalarFunction]

ScalarToMonad = Callable[[Scalar], "Monad"]


class Monad(ABC):
    @classmethod
    @abstractmethod
    def unit(cls, value: Any) -> "Monad":
        raise NotImplementedError

    @abstractmethod
    def map(self, function: RegularFunction) -> "Monad":
        raise NotImplementedError

    @abstractmethod
    def apply(self, lifted: "Monad") -> "Monad":
        raise NotImplementedError

    @abstractmethod
    def bind(self, function: Callable[[Scalar], "Monad"]) -> "Monad":
        raise NotImplementedError


@dataclass
class Identity(Monad):
    """
    The identity monad. It does nothing but wrap a value.
    """

    value: Union[Scalar, Callable]

    @classmethod
    def unit(cls, value: Any) -> "Monad":
        return unit(value, cls)

    def map(self, function: RegularFunction) -> "Monad":
        return map(self, function)

    def apply(self, lifted: "Monad") -> "Monad":
        return apply(self, lifted)

    def bind(self, function: Callable[[Scalar], "Monad"]) -> "Monad":
        return bind(self, function)

    def __eq__(self, other: "Monad"):

        if self.value == other.value:
            return True
        elif self.value is other.value:
            return True
        elif callable(self.value) and callable(other.value):
            # we assume both functions accept integers for simplicity's sake
            i = random.randrange(0, 100)
            return self.value(i) == other.value(i)
        else:
            return False


def unit(
    value: Union[Scalar, RegularFunction], M: Type[Monad] = Identity
) -> Monad:
    """
    AKA: return, pure, yield, point
    """
    return M(value) if not isinstance(value, M) else value


def map(monad: Monad, function: RegularFunction) -> Monad:

    if not callable(monad.value):
        return monad.unit(function(monad.value))
    else:
        return monad.unit(partial_or_composition(function, monad.value))


def partial_or_composition(function, wrapped_function):
    try:
        return function(wrapped_function)
    except TypeError:
        return compose(wrapped_function, function)


def apply(lifted_function: Monad, lifted: Monad) -> Monad:
    """AKA: ap, <*>"""
    lifted_function.value: RegularFunction

    return map(lifted, lifted_function.value)


def bind(monad: Monad, function: Callable[[Scalar], Monad]) -> Monad:
    """AKA: flatMap, andThen, collect, SelectMany, >>=, =<<"""
    return (
        function(monad.value)
        if not callable(monad.value)
        else map(monad.unit(monad.value), function)
    )


# ---- tests ---- #


@st.composite
def monads(draw):

    # scalars = st.one_of(
    #     st.integers(), st.floats(allow_nan=False), st.text(), st.booleans()
    # )

    scalars = st.integers()

    unary_functions = st.functions(like=lambda x: x, returns=scalars)

    value = draw(st.one_of(scalars, unary_functions))

    value = value if not callable(value) else memoize(value)

    return Identity(value)


@given(monad=monads(), integer=st.integers(), f=infer, g=infer)
def test_map(
    monad: Monad,
    integer: int,
    f: Callable[[int, int], int],
    g: Callable[[int, int], int],
):
    """
    fmap id  ==  id
    fmap (f . g)  ==  fmap f . fmap g
    """
    # make our generated function `f deterministic

    f, g = memoize(f), memoize(g)

    monad = unit(integer)

    assert map(monad, identity) == monad
    # method form
    assert monad.map(identity) == monad

    # composition

    assert map(unit(integer), compose(f, g)) == map(map(unit(integer), g), f)
    # method form
    assert unit(integer).map(compose(f, g)) == unit(integer).map(g).map(f)


@settings(report_multiple_bugs=False)
@given(monad=monads(), value=infer, f=infer, g=infer)
def test_bind(
    monad: Monad, value: Scalar, f: RegularFunction, g: RegularFunction
):
    """
    unit(a) >>= λx → f(x) ↔ f(a)
    ma >>= λx → unit(x) ↔ ma
    ma >>= λx → (f(x) >>= λy → g(y)) ↔ (ma >>= λx → f(x)) >>= λy → g(y)
    """
    f, g = _modify(f), _modify(g)

    # left identity

    assert bind(unit(value), f) == f(value)
    # method form
    assert unit(value).bind(f) == f(value)

    # right identity

    assert bind(monad, unit) == monad
    # method form
    assert monad.bind(unit) == monad

    # associativity

    assert bind(bind(monad, f), g) == bind(monad, lambda x: bind(f(x), g))
    # method syntax
    assert monad.bind(f).bind(g) == monad.bind(lambda x: bind(f(x), g))


@settings(report_multiple_bugs=False)
@given(
    monad=monads(),
    other_monad=monads(),
    integer=st.integers(),
    f=infer,
    g=infer,
)
def test_app(
    monad, other_monad, integer, f: RegularFunction, g: RegularFunction
):
    """
    identity

        pure id <*> v = v

    homomorphism

        pure f <*> pure x = pure (f x)

    interchange

        u <*> pure y = pure ($ y) <*> u

    composition

        pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
    """

    # f, g = monad.unit(determinize(f)), monad.unit(determinize(g))

    f, g = memoize(f), memoize(g)

    # identity

    assert apply(unit(identity), monad) == monad
    # method syntax
    assert unit(identity).apply(monad) == monad

    """
    homomorphism

        pure f <*> pure x = pure (f x)
    """

    assert apply(unit(f), monad) == unit(f(monad.value))

    assert unit(f).apply(monad) == unit(f(monad.value))

    """
    interchange

        u <*> pure y = pure ($ y) <*> u
    
    """

    u = unit(identity)  # todo: make this generic f

    y = integer

    l = apply(u, unit(y))
    r = apply(unit(lambda g: g(y)), u)
    assert l == r, f"{l} != {r}"
    # method form
    assert u.apply(unit(y)) == unit(lambda g: g(y)).apply(u)

    """
    The final applicative law mimics the second functor law. 
    It is a composition law. 
    It states that function composition holds 
    across applications within the functor

        pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
    """
    # v, w = monad, other_monad
    # v, w = unit(f), unit(g)

    # u = unit(lambda x: x + 1)
    # v = unit(lambda x: x * 2)
    # w = unit(lambda x: x + 3)
    # left = unit(compose).apply(u).apply(v).apply(w)
    # right = u.apply(v.apply(w))
    # print(left.value, right.value)
    # assert left == right, f"{left} != {right}"


def _modify(function: RegularFunction):
    """Memoize function and wrap it in a monad."""

    @memoize
    def f(x):

        return unit(function(x))

    return f


def compose(f: RegularFunction):
    """
    Compose two functions together in curried form.
    """

    def i(g: RegularFunction):
        def j(x: Scalar):
            return f(g(x))

        return j

    return i


def memoize(func):
    return lru_cache(maxsize=None)(func)


def compose(f, g):
    def f_after_g(x):
        return f(g(x))

    return f_after_g


def identity(x: Any) -> Any:
    return x


def test():
    test_map()
    test_bind()
    test_app()


if __name__ == "__main__":
    test()
