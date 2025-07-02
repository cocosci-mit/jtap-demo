import jax
import jax.numpy as jnp
from genjax import ExactDensity, Pytree
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

@Pytree.dataclass
class UniformPosition2D(ExactDensity):
    def sample(self, key, x_min, x_max, y_min, y_max, **kwargs):
        x_key, y_key = jax.random.split(key, 2)
        x = tfd.Uniform(x_min, x_max).sample(seed = x_key)
        y = tfd.Uniform(y_min, y_max).sample(seed = y_key)
        return jnp.array([x, y])

    def logpdf(self, v, x_min, x_max, y_min, y_max, **kwargs):
        return jnp.log(1/(x_max - x_min)) + jnp.log(1/(y_max - y_min))

@Pytree.dataclass
class TruncatedNormPosition2D(ExactDensity):
    def sample(self, key, x_params, y_params, **kwargs):
        x_mean, x_scale, x_min, x_max = x_params
        y_mean, y_scale, y_min, y_max = y_params
        x, y = tfd.TruncatedNormal(loc=[x_mean, y_mean], scale=[x_scale, y_scale], 
            low=[x_min, y_min], high=[x_max, y_max]).sample(seed = key)
        return jnp.array([x, y])

    def logpdf(self, v, x_params, y_params, **kwargs):
        x_mean, x_scale, x_min, x_max = x_params
        y_mean, y_scale, y_min, y_max = y_params
        return tfd.TruncatedNormal(loc=[x_mean, y_mean], scale=[x_scale, y_scale], 
            low=[x_min, y_min], high=[x_max, y_max]).log_prob(v)    
    
uniformposition2d = UniformPosition2D()
truncatednormposition2d = TruncatedNormPosition2D()

@Pytree.dataclass
class CircularNormal(ExactDensity):
    # in radians
    def sample(self, key, dir_mean, dir_scale, **kwargs):
        # we assume that the sampled angle can maximally move 2*pi in either direction (hence the truncation)
        sampled_dir = tfd.TruncatedNormal(loc=dir_mean, scale=dir_scale, low = dir_mean - 2*jnp.pi, high = dir_mean + 2*jnp.pi).sample(seed = key) 
        return jnp.mod(sampled_dir + jnp.pi,2*jnp.pi) - jnp.pi

    def logpdf(self, v, dir_mean, dir_scale, **kwargs):
        # 2 places where there is weight (one from each direction)
        dist = tfd.TruncatedNormal(loc=0.0, scale=dir_scale, low = -2*jnp.pi, high = 2*jnp.pi)
        smallest_angular_diff = jnp.abs(jnp.mod(v - dir_mean + jnp.pi, 2 * jnp.pi)- jnp.pi)
        # from nearer direction as well as further direction (weights on both sides)
        return jnp.log(dist.prob(smallest_angular_diff) + dist.prob(2*jnp.pi - smallest_angular_diff)) 
    
circular_normal = CircularNormal()