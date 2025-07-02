import jax
import jax.numpy as jnp
from genjax import gen
from genjax import ChoiceMapBuilder as C
from jtap.distributions import *
from jtap.model import *

@jax.jit
def determine_occupied_cells(carry, masked_barrier):
    # includes partial blockage
    occupied_arr, x_grid, y_grid, size = carry
    barrier_x, barrier_y, barrier_size_x, barrier_size_y = masked_barrier.value
    occupied_arr = jax.lax.select(
        masked_barrier.flag,
        jnp.logical_or(occupied_arr,
            (x_grid >= (barrier_x - size)) * 
            (x_grid < (barrier_x + barrier_size_x)) * 
            (y_grid >= (barrier_y - size)) * 
            (y_grid < (barrier_y + barrier_size_y))
        ),
        occupied_arr    
    )
    return (occupied_arr, x_grid, y_grid, size), None
# PROPOSAL SPECIFIC FUNCTIONS

@jax.jit
def determine_hidden_cells(carry, masked_occluder):
    # includes partial hidden
    hidden_arr, x_grid, y_grid, size = carry
    occ_x, occ_y, occ_size_x, occ_size_y = masked_occluder.value
    hidden_arr = jax.lax.select(
        masked_occluder.flag,
        jnp.logical_or(hidden_arr,
            ((x_grid >= occ_x) * (x_grid <= occ_x + occ_size_x - size)) *
            ((y_grid >= occ_y) * (y_grid <= occ_y + occ_size_y - size))
        ),
        hidden_arr    
    )
    return (hidden_arr, x_grid, y_grid, size), None

def data_driven_size_and_position(obs, scale):
    num_obj_pixels = jnp.sum(obs == jnp.int8(2))
    obj_bool_any_axis0 = jnp.any(obs == jnp.int8(2), axis=0)
    obj_bool_any_axis1 = jnp.any(obs == jnp.int8(2), axis=1)

    leftmost_target = scale * jnp.argmin(jnp.where(obj_bool_any_axis1, jnp.arange(obs.shape[0]), jnp.float32(jnp.inf)))
    rightmost_target = scale + scale * jnp.argmax(jnp.where(obj_bool_any_axis1, jnp.arange(obs.shape[0]), -jnp.float32(jnp.inf)))
    bottommost_target = scale * jnp.argmin(jnp.where(obj_bool_any_axis0, jnp.arange(obs.shape[1]), jnp.float32(jnp.inf)))
    topmost_target = scale + scale * jnp.argmax(jnp.where(obj_bool_any_axis0, jnp.arange(obs.shape[1]), -jnp.float32(jnp.inf)))
    mean_size = 0.5 * (rightmost_target - leftmost_target) + 0.5 * (topmost_target - bottommost_target)
    
    x_prop, y_prop = leftmost_target, bottommost_target
    is_fully_hidden = jnp.logical_not(jnp.logical_or(jnp.any(obj_bool_any_axis0), jnp.any(obj_bool_any_axis1)))

    return mean_size, x_prop, y_prop, is_fully_hidden, num_obj_pixels

@gen
def visible_target_proposal(proposal_args, pos_noise, size_noise):
    mi, _, _, obs = proposal_args
    scale = mi.image_discretization

    mean_size, x_prop, y_prop, *_ = data_driven_size_and_position(obs, scale)

    size = genjax.truncated_normal(mean_size, size_noise, *mi.size_lims) @ "size"
    x, y = truncatednormposition2d((x_prop, pos_noise, 0., mi.scene_dim[0] - size), 
        (y_prop, pos_noise, 0.,  mi.scene_dim[1] - size)) @ "xy" # joint sampling of x and y

    return size, x, y, x_prop, y_prop


@gen
def hidden_target_proposal(proposal_args, size_noise):
    # As we don't know the size, assume object is a big as possible
    mi, masked_occluders, masked_barriers, _ = proposal_args
    x_grid, y_grid =  jnp.meshgrid(mi.pix_x, mi.pix_y, indexing='ij')

    hidden_arr = jnp.zeros((mi.pix_x.shape[0],mi.pix_y.shape[0])).astype(jnp.bool)
    (hidden_arr, *_), _ = jax.lax.scan(determine_hidden_cells, (hidden_arr, x_grid, y_grid, mi.size_lims[1]), masked_occluders)
    hidden_probs = hidden_arr/hidden_arr.sum()

    occupied_arr = jnp.zeros(hidden_arr.shape).astype(jnp.bool)
    (occupied_arr, *_), _ = jax.lax.scan(determine_occupied_cells, (occupied_arr, x_grid, y_grid, mi.size_lims[1]), masked_barriers)
    unoccupied_arr = jnp.invert(occupied_arr).astype(jnp.int8)
    xy_probs = hidden_probs * unoccupied_arr
    xy_probs = xy_probs/xy_probs.sum()

    # NOTE: Size should be uniform, here i am setting to mean and  variance of 1,0.2
    size = genjax.truncated_normal(jnp.float32(1.), size_noise, *mi.size_lims) @ "size"
    x, y = initialposition2d(xy_probs, x_grid, y_grid) @ "xy" # joint sampling of x and y

    return size, x, y, x, y


target_proposal_switch = genjax.switch(
    visible_target_proposal,
    hidden_target_proposal
)

@gen
def init_proposal(init_proposal_args):
    mi, *_, first_frame = init_proposal_args

    size_noise = mi.σ_size_initprop
    pos_noise = mi.σ_pos_initprop

    num_target_pixels = jnp.sum(first_frame == jnp.int8(2))
    obj_is_fully_hidden = jnp.equal(num_target_pixels, jnp.int32(0))

    # have noise around a fixed size (1.0) if object is fully hidden
    size_noise = jnp.where(obj_is_fully_hidden, jnp.float32(0.2), size_noise)

    switch_branch = jnp.where(obj_is_fully_hidden, jnp.int32(1), jnp.int32(0))

    size, x, y, prop_x, prop_y = target_proposal_switch(
        switch_branch,
        (init_proposal_args, pos_noise, size_noise),
        (init_proposal_args, size_noise)
    ) @ 'prop_switch'
    
    return {
        "proposal_branch": switch_branch,
        "proposed_size": size,
        "proposed_x": x,
        "proposed_y": y,
        "presample_proposed_x": prop_x,
        "presample_proposed_y": prop_y
    }

@jax.jit
def init_choicemap_translator(cm_proposed, cm_original):
    switch_prop = cm_proposed('prop_switch')
    
    xy_prop = switch_prop.get_submap('xy').extend('xy')
    size_prop = switch_prop.get_submap('size').extend('size')

    cmprop = xy_prop.merge(size_prop)

    return C.d(
            {
                'init' : cm_original('init').merge(cmprop),
                'init_obs' : cm_original['init_obs']
            }
        )
