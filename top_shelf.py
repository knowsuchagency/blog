import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, lru_cache, wraps
from typing import *

import hypothesis.strategies as st
from hypothesis import given, infer

Scalar = Union[AnyStr, int, bool]

RegularFunction = Callable[[Scalar], Scalar]


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

        if callable(self.value) and callable(other.value):
            # we assume both functions accept integers for simplicity's sake
            i = random.randrange(0, 100)
            return self.value(i) == other.value(i)

        return self.value == other.value


def unit(value: Scalar, M: Type[Monad] = Identity) -> Monad:
    """
    AKA: return, pure, yield, point
    """
    return M(value) if not isinstance(value, M) else value


def map(monad: Monad, function: RegularFunction) -> Monad:
    """AKA: fmap, lift, Select"""
    return monad.unit(
        function(monad.value)
        if not callable(monad.value)
        else lambda x: monad.value(function(x))
    )


def apply(lifted_function: Monad, lifted: Monad) -> Monad:
    """AKA: ap, <*>"""
    lifted_function.value: RegularFunction

    return map(lifted, lifted_function.value)


def bind(monad: Monad, function: Callable[[Scalar], Monad]) -> Monad:
    """AKA: flatMap, andThen, collect, SelectMany, >>=, =<<"""
    return function(monad.value)


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

    f = memoize(f)

    assert map(monad, identity) == monad
    # method form
    assert monad.map(identity) == monad

    f = partial(f, integer)

    g = partial(f, integer)

    f_after_g = lambda x: f(g(x))

    m = unit(integer)

    assert map(m, f_after_g) == map(map(m, g), f)
    # method form
    assert m.map(f_after_g) == m.map(g).map(f)


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


@given(monad=monads(), integer=st.integers(), f=infer, g=infer)
def test_app(
    monad, integer, f: Callable[[int], int], g: Callable[[int, int], int]
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

    f = memoize(f)

    # identity

    assert apply(unit(identity), monad) == monad
    # method syntax
    assert unit(identity).apply(monad) == monad

    """
    homomorphism

        pure f <*> pure x = pure (f x)
    """

    m = unit(integer)

    assert apply(unit(f), m) == unit(f(m.value))

    assert unit(f).apply(m) == unit(f(m.value))

    """
    The third law is the interchange law. 
    It’s a little more complicated, so don’t sweat it too much. 
    It states that the order that we wrap things shouldn’t matter. 
    One on side, we apply any applicative over a pure wrapped object. 
    On the other side, first we wrap a function applying the object as an argument. 
    Then we apply this to the first applicative. These should be the same.

        u <*> pure y = pure ($ y) <*> u
    
    """

    # assert m.apply(monad.unit(f)) == monad.unit(lambda x: f(x)).apply(f)

    # composition

    # m = monad.unit(identity)
    # m = monad
    #
    # left = apply(apply(apply(monad.unit(compose), m), f), g)
    # right = apply(m, apply(f, g))
    # assert left == right, f'{left} != {right} ; {left.value(1)}'


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
    @lru_cache(maxsize=None)
    @wraps(func)
    def inner(*args, **kwargs):
        return func(*args, **kwargs)

    return inner


def identity(x: Any) -> Any:
    return x


def test():
    test_map()
    test_bind()
    test_app()


if __name__ == "__main__":
    test()
