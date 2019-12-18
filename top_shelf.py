import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, lru_cache
from typing import *

import hypothesis.strategies as st
from hypothesis import given, infer, settings

Scalar = Union[AnyStr, int, bool]

ScalarToScalar = Callable[[Scalar], Scalar]

ScalarOrScalarFunction = Union[Scalar, ScalarToScalar]

RegularFunction = Callable[[ScalarOrScalarFunction], ScalarOrScalarFunction]


class Functor(ABC):
    @classmethod
    @abstractmethod
    def unit(cls, value: Any) -> "Functor":
        raise NotImplementedError

    @abstractmethod
    def map(self, function: RegularFunction) -> "Functor":
        raise NotImplementedError


class Applicative(Functor):
    @abstractmethod
    def apply(self, lifted: "Applicative") -> "Applicative":
        raise NotImplementedError


class Monad(Applicative):
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
    def unit(cls, value: Any) -> "Identity":
        return unit(value, cls)

    def map(self, function: RegularFunction) -> "Identity":
        return map(self, function)

    def apply(self, lifted: "Identity") -> "Identity":
        return apply(self, lifted)

    def bind(self, function: Callable[[Scalar], "Identity"]) -> "Identity":
        return bind(self, function)

    def __eq__(self, other: Any):
        if not isinstance(other, Identity):
            return False
        elif self.value is other.value:
            return True
        elif self.value == other.value:
            return True
        elif callable(self.value) and callable(other.value):
            # we assume both functions accept 0 for simplicity's sake
            return self.value(0) == other.value(0)
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
    """
    Given a monad and a function, return a new monad where the function is applied to the monad's value.

    Note, it's normally bad practice to define control flow using exceptions, but due to Python's dynamic
    nature, we aren't guaranteed to have the necessary type information up-front in order to know whether
    we need to simply apply the function to the wrapped value, partially apply two functions, or compose them.

    Without type-annotated arguments, we're left to figure out the control flow ourselves through experimentation.

    """

    try:
        return monad.unit(function(monad.value))
    except TypeError:
        return monad.unit(partial_or_composition(function, monad.value))


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

    scalars = st.integers()

    unary_functions = st.functions(like=lambda x: x, returns=scalars)

    value = draw(st.one_of(scalars, unary_functions))

    value = value if not callable(value) else memoize(value)

    return Identity(value)


@given(integer=st.integers(), f=infer, g=infer)
def test_map(
    integer: int, f: Callable[[int, int], int], g: Callable[[int, int], int]
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
    f, g = _memoize_and_monadify(f), _memoize_and_monadify(g)

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
    integer=st.integers(),
    f=infer,
    g=infer,
    u=infer,
    v=infer,
    w=infer,
)
def test_app(
    monad,
    integer,
    f: RegularFunction,
    g: RegularFunction,
    u: RegularFunction,
    v: RegularFunction,
    w: RegularFunction,
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

    f, g, u, v, w = memoize(f), memoize(g), memoize(u), memoize(v), memoize(w)

    # identity

    assert apply(unit(identity), monad) == monad
    # method syntax
    assert unit(identity).apply(monad) == monad

    """
    homomorphism

        pure f <*> pure x = pure (f x)
    """
    x = integer

    assert apply(unit(f), unit(x)) == unit(f(x))

    assert unit(f).apply(unit(x)) == unit(f(x))

    """
    interchange

        u <*> pure y = pure ($ y) <*> u
    
    """

    y = integer

    assert apply(unit(f), unit(y)) == apply(unit(lambda g: g(y)), unit(f))
    # method form
    assert unit(f).apply(unit(y)) == unit(lambda g: g(y)).apply(unit(f))

    """
    composition

        pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
    """

    u, v, w = unit(u), unit(v), unit(w)

    assert unit(compose).apply(u).apply(v).apply(w) == u.apply(v.apply(w))


def _memoize_and_monadify(function: RegularFunction):
    """Memoize function and wrap its return value in a monad."""

    @memoize
    def f(x):

        return unit(function(x))

    return f


def memoize(func):
    return lru_cache(maxsize=None)(func)


def identity(x: Any) -> Any:
    return x


def partial_or_composition(f: RegularFunction, g: RegularFunction):
    if len(inspect.signature(f).parameters) > 1:
        # f is probably the composition function
        return partial(f, g)
    else:
        return compose(f, g)


def compose(f, g):
    def f_after_g(x):
        return f(g(x))

    return f_after_g


def test():
    test_map()
    test_bind()
    test_app()


if __name__ == "__main__":
    test()
