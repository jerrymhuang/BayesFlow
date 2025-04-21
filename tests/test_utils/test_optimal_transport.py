import keras
import pytest

from bayesflow.utils import optimal_transport
from tests.utils import assert_allclose


@pytest.mark.jax
def test_jit_compile():
    import jax

    x = keras.random.normal((128, 8), seed=0)
    y = keras.random.normal((128, 8), seed=1)

    ot = jax.jit(optimal_transport, static_argnames=["regularization", "seed"])
    ot(x, y, regularization=1.0, seed=0, max_steps=10)


def test_shapes():
    x = keras.random.normal((128, 8), seed=0)
    y = keras.random.normal((128, 8), seed=1)

    ox, oy = optimal_transport(x, y, regularization=1.0, seed=0, max_steps=10)

    assert keras.ops.shape(ox) == keras.ops.shape(x)
    assert keras.ops.shape(oy) == keras.ops.shape(y)


def test_transport_cost_improves():
    x = keras.random.normal((1024, 2), seed=0)
    y = keras.random.normal((1024, 2), seed=1)

    before_cost = keras.ops.sum(keras.ops.norm(x - y, axis=-1))

    x, y = optimal_transport(x, y, regularization=0.1, seed=0, max_steps=None)

    after_cost = keras.ops.sum(keras.ops.norm(x - y, axis=-1))

    assert after_cost < before_cost


def test_assignment_is_optimal():
    x = keras.ops.convert_to_tensor(
        [
            [-1, 2],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [-1, -2],
        ]
    )
    optimal_y = keras.ops.convert_to_tensor(
        [
            [1, 2],
            [1, 1],
            [1, 0],
            [1, -1],
            [1, -2],
        ]
    )
    y = keras.random.shuffle(optimal_y, axis=0, seed=0)

    x, y = optimal_transport(x, y, regularization=0.1, seed=0, max_steps=None, scale_regularization=False)

    assert_allclose(x, y)
