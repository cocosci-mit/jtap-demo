from typing import Union, TypeVar
import numpy as np
import jax.numpy as jnp

T = TypeVar('T', bound=Union[np.ndarray, jnp.ndarray, list, tuple, int, float, bool])

def i_(x: T) -> jnp.ndarray:
    return jnp.int32(x)

def i8_(x: T) -> jnp.ndarray:
    return jnp.int8(x)

def f_(x: T) -> jnp.ndarray:
    return jnp.float32(x)

def b_(x: T) -> jnp.ndarray:
    return jnp.bool_(x)

def a_(x: T) -> jnp.ndarray:
    return jnp.array(x)