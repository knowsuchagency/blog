import functools
import math
from dataclasses import dataclass
from numbers import Number, Complex
from typing import *

import hypothesis.strategies as st
from hypothesis import given, infer

Scalar = Union[AnyStr, Number, bool]

RegularFunction = Callable[[Scalar], Scalar]


def identity(x: Any) -> Any:
    return x


def unit(M: Type["Monad"], value: Scalar) -> "Monad":
    """
    AKA: return, pure, yield, point
    """
    return M(value)


def map(monad: "Monad", function: RegularFunction) -> "Monad":
    """AKA: fmap, lift, Select"""
    return monad.unit(function(monad.value))


def apply(lifted_function: "Monad", lifted_value: "Monad") -> "Monad":
    """AKA: ap, <*>"""
    lifted_function.value: RegularFunction

    # since python doesn't curry functions automatically,
    # we have to do it ourselves in this case to obey the
    # applicative laws

    if callable(lifted_value.value):
        g = lifted_value.value
        lifted_value = lifted_value.unit(lambda f: g(f))

    return map(lifted_value, lifted_function.value)


def bind(monad: "Monad", function: Callable[[Scalar], "Monad"]) -> "Monad":
    """AKA: flatMap, andThen, collect, SelectMany, >>=, =<<"""
    return function(monad.value)


@dataclass
class Monad:
    """
    The identity monad. It does nothing but wrap a value.
    """

    value: Scalar

    @classmethod
    def unit(cls, value: Any) -> "Monad":
        return unit(cls, value) if not isinstance(value, cls) else value

    def map(self, function: RegularFunction) -> "Monad":
        return map(self, function)

    def apply(self, lifted_function: "Monad") -> "Monad":
        return apply(self, lifted_function)

    def bind(self, function: Callable[[Scalar], "Monad"]) -> "Monad":
        return bind(self, function)


# ---- tests ---- #


scalars = lambda: st.one_of(
    st.integers(), st.floats(allow_nan=False), st.text(), st.booleans()
)


@st.composite
def monads(draw) -> Monad:
    value = draw(scalars())
    return Monad(value)


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

    f = functools.lru_cache(maxsize=None)(f)

    assert map(monad, identity) == monad
    # method form
    assert monad.map(identity) == monad

    f = functools.partial(f, integer)

    g = functools.partial(f, integer)

    f_after_g = lambda x: f(g(x))

    m = monad.unit(integer)

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

    assert bind(unit(Monad, value), f) == f(value)
    # method form
    assert Monad.unit(value).bind(f) == f(value)

    # right identity

    assert bind(monad, monad.unit) == monad
    # method form
    assert monad.bind(monad.unit) == monad

    # associativity

    assert bind(bind(monad, f), g) == bind(monad, lambda x: bind(f(x), g))
    # method syntax
    assert monad.bind(f).bind(g) == monad.bind(lambda x: bind(f(x), g))


@given(monad=monads(), f=infer, g=infer)
def test_app(monad, f: RegularFunction, g: RegularFunction):

    determinize = functools.lru_cache(maxsize=None)

    f, g = monad.unit(determinize(f)), monad.unit(determinize(g))

    # identity

    assert apply(monad.unit(identity), monad) == monad
    # method syntax
    assert monad.unit(identity).apply(monad) == monad

    # composition

    m = monad.unit(identity)

    apply(apply(apply(monad.unit(compose), m), f), g) == apply(m, apply(f, g))


def _modify(function: RegularFunction):
    """Wrap function in a monad, make it deterministic, and avoid NaN since we can't check for equality with it."""

    @functools.lru_cache(maxsize=None)
    def f(x):

        result = Monad(function(x))

        if isinstance(result.value, Complex):
            if math.isnan(result.value.imag) or math.isnan(result.value.real):
                return Monad(None)
        elif isinstance(result.value, Number) and math.isnan(result.value):
            return Monad(None)

        return result

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


def test():
    test_map()
    test_bind()
    test_app()


if __name__ == "__main__":
    test()
