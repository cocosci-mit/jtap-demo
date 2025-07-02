import jax
import jax.numpy as jnp
from genjax import ExactDensity, Pytree
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
    
""" Distribution Utilities """ 

unicat = lambda x: jnp.ones(len(x)) / len(x)
normalize = lambda x: x / jnp.sum(x)

""" Define Distributions """

@Pytree.dataclass
class LabeledCategorical(ExactDensity):
    def sample(self, key, probs, labels, **kwargs):
        cat = tfd.Categorical(probs=normalize(probs))
        cat_index = cat.sample(seed=key)
        return labels[cat_index]

    def logpdf(self, v, probs, labels, **kwargs):
        w = jnp.log(jnp.sum(normalize(probs) * (labels==v)))
        return w

@Pytree.dataclass
class UniformCategorical(ExactDensity):
    def sample(self, key, labels, **kwargs):
        cat = tfd.Categorical(probs=jnp.ones(len(labels)) / len(labels))
        cat_index = cat.sample(seed=key)
        return labels[cat_index]

    def logpdf(self, v, labels, **kwargs):
        probs = jnp.ones(len(labels)) / len(labels)
        logpdf = jnp.log(probs)
        w = logpdf[0]
        return w
    
@Pytree.dataclass
class InitialPosition2D(ExactDensity):
    def sample(self, key, probs_2d, x_grid, y_grid, **kwargs):
        probs_1d = probs_2d.flatten()
        cat = tfd.Categorical(probs=normalize(probs_1d))
        cat_index = cat.sample(seed=key)
        x_idx, y_idx = jnp.unravel_index(cat_index, probs_2d.shape)
        return jnp.array([x_grid[x_idx, y_idx], y_grid[x_idx, y_idx]])

    def logpdf(self, v, probs_2d, x_grid, y_grid, **kwargs):
        probs_1d = probs_2d.flatten()
        x_position, y_position = v
        x_idx = jnp.argmin(jnp.abs(x_grid[:, 0] - x_position))
        y_idx = jnp.argmin(jnp.abs(y_grid[0, :] - y_position))
        cat_idx = jnp.ravel_multi_index((x_idx, y_idx), probs_2d.shape, mode='wrap')
        one_hot_cat = jnp.zeros(probs_1d.shape).at[cat_idx].set(1).astype(jnp.bool)
        w = jnp.log(jnp.sum(normalize(probs_1d) * one_hot_cat))
        return w

initialposition2d = InitialPosition2D()
labcat = LabeledCategorical()
uniformcat = UniformCategorical()
find_nearest = jax.jit(lambda A, target : A[jnp.argmin(jnp.abs(A - target))])
find_nearest_valid = jax.jit(lambda A, target, valid : A[jnp.argmin(jnp.abs(jnp.where(valid,A,jnp.inf) - target))])

# probably a good idea to eventually allow in non-equi distant domain values. 
def discrete_norm(μ, σ, dom, fit_to_discrete = False):
    μ = jax.lax.select(
        fit_to_discrete,
        find_nearest(dom, μ),
        μ
    )
    div = (dom[1] - dom[0]) / 2.
    unnormed_dist = jnp.nan_to_num(normalize(jax.vmap(lambda i: tfd.Normal(loc=μ, scale=σ).cdf(i + div) - tfd.Normal(loc=μ, scale=σ).cdf(i - div))(dom)))
    dist = ((μ < dom[0]) * (1-jnp.sum(unnormed_dist))) * jnp.hstack((jnp.array([1]), jnp.zeros(len(dom)-1))) + (
        (μ > dom[-1]) * (1-jnp.sum(unnormed_dist))) * jnp.hstack((jnp.zeros(len(dom)-1), jnp.array([1]))) + unnormed_dist
    return dist

def discrete_2dnorm(μ_x, σ_x, dom_x, μ_y, σ_y, dom_y, fit_to_discrete=False):
    # Create discrete normal distributions for each dimension
    dist_x = discrete_norm(μ_x, σ_x, dom_x, fit_to_discrete=fit_to_discrete)
    dist_y = discrete_norm(μ_y, σ_y, dom_y, fit_to_discrete=fit_to_discrete)
    dist_2d = jnp.outer(dist_x, dist_y)

    return dist_2d

def discrete_vonmises(μ, k, dom, fit_to_discrete = False):
    μ = jax.lax.select(
        fit_to_discrete,
        find_nearest(dom, μ),
        μ
    )
    div = (dom[1] - dom[0]) / 2.
    return jnp.nan_to_num(normalize(jax.vmap(lambda i: tfd.VonMises(loc=μ, concentration=k).cdf(i + div) - tfd.VonMises(loc=μ, concentration=k).cdf(i - div))(dom)))
