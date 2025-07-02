import jax
import jax.numpy as jnp
from genjax import Pytree, ExactDensity
from jtap.utils import slice_pt

@Pytree.dataclass
class MentalPhysicsLikelihood(ExactDensity):
    def sample(self, key, render_args, flip_prob, *args, **kwargs):
        return render_scene(*render_args)

    def logpdf(self, obs_image, render_args, flip_prob, *args, **kwargs):
        return jnp.sum(
                # jnp.where(obs_image == render_scene(*render_args), jnp.log(1 - flip_prob), jnp.log(flip_prob/5))
                jnp.where(obs_image == render_scene(*render_args), jnp.log(1 - flip_prob), jnp.log(flip_prob))
        )
    
likelihood_model = MentalPhysicsLikelihood()


def log_gaussian_kernel(size: int, sigma: float) -> jnp.ndarray:
    ax = jnp.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = jnp.meshgrid(ax, ax)
    kernel = -(xx**2 + yy**2) / (2.0 * sigma**2)
    kernel = kernel - jax.nn.logsumexp(kernel) # normalize
    return kernel

@jax.jit
def render_scene(pix_x, pix_y, shape, size, x, y, masked_barriers, masked_occluders, red_sensor, green_sensor):
    # Precompute the grid
    x_vals, y_vals = jnp.meshgrid(pix_x, pix_y, indexing='ij')
    max_barriers = masked_barriers.flag.shape[0]
    max_occluders = masked_occluders.flag.shape[0]

    # Initialize the image
    image = jnp.zeros((pix_x.shape[0], pix_y.shape[0]), dtype=jnp.int8)

    # Render green and red sensors

    sensor_x, sensor_y, sensor_size_x, sensor_size_y = green_sensor
    image = jnp.where((x_vals >= sensor_x) & (y_vals >= sensor_y) & (x_vals < sensor_x + sensor_size_x) & (y_vals < sensor_y + sensor_size_y), jnp.int8(5), image)

    sensor_x, sensor_y, sensor_size_x, sensor_size_y = red_sensor
    image = jnp.where((x_vals >= sensor_x) & (y_vals >= sensor_y) & (x_vals < sensor_x + sensor_size_x) & (y_vals < sensor_y + sensor_size_y), jnp.int8(4), image)

    for i in range(max_barriers):
        barrier_x, barrier_y, barrier_size_x, barrier_size_y = slice_pt(masked_barriers.value,i)
        image = jnp.where(masked_barriers.flag[i] & (x_vals >= barrier_x) & (y_vals >= barrier_y) & (x_vals < barrier_x + barrier_size_x) & (y_vals < barrier_y + barrier_size_y), jnp.int8(3), image)

    # Render the target
    r = size / 2
    # No squre allowed
    # image = jax.lax.select(
    #     shape == 0,
    #     jnp.where((x_vals >= x) & (y_vals >= y) & (x_vals < x + size) & (y_vals < y + size), 2, image),
    #     jnp.where(((x_vals - (x + r))**2 + (y_vals - (y + r))**2) <= r**2, 2, image)
    # )

    image = jnp.where(((x_vals - (x + r))**2 + (y_vals - (y + r))**2) <= r**2, jnp.int8(2), image)

    for i in range(max_occluders):
        occluder_x, occluder_y, occluder_size_x, occluder_size_y = slice_pt(masked_occluders.value,i)
        image = jnp.where(masked_occluders.flag[i] & (x_vals >= occluder_x) & (y_vals >= occluder_y) & (x_vals < occluder_x + occluder_size_x) & (y_vals < occluder_y + occluder_size_y), jnp.int8(1), image)

    return image