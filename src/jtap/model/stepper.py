import jax
import jax.numpy as jnp
import genjax
from genjax import gen
from jtap.distributions import circular_normal
from .collision import velocity_transform
from jtap.utils import ModelOutput

@gen
def stepper_model(mo, mi, inference_mode_bool):

    new_size = mo.size

    next_speed, speed_noise, next_dir, dir_σ, next_x, next_y, \
    collision_branch, is_target_hidden, is_target_visible, \
        is_target_partially_hidden, stopped_early = velocity_transform(mo, mi, mo.friction, mo.elasticity, new_size, inference_mode_bool)

    new_speed = genjax.truncated_normal(next_speed, speed_noise, jnp.float32(0.), mi.max_speed) @ "speed"
    new_dir = circular_normal(next_dir, dir_σ) @ "dir"
    pos_noise = jnp.where(inference_mode_bool, mi.σ_pos, mi.σ_pos_sim)

    new_x = genjax.truncated_normal(next_x, pos_noise, jnp.float32(0.), mi.scene_dim[0] - new_size) @ 'x'
    new_y = genjax.truncated_normal(next_y, pos_noise, jnp.float32(0.), mi.scene_dim[1] - new_size) @ 'y'

    last_collision_data = jnp.where(
        jnp.equal(collision_branch, jnp.float32(4)),
        mo.last_collision_data,
        jnp.array([mo.T + jnp.int32(1), new_x, new_y, new_speed, new_dir])
    )

    return ModelOutput(
        shape=mo.shape, size=new_size, x=new_x, y=new_y, speed=new_speed, dir=new_dir, 
        num_occs=mo.num_occs, num_barriers=mo.num_barriers, masked_occluders=mo.masked_occluders, 
        masked_barriers=mo.masked_barriers, is_target_hidden=is_target_hidden, 
        is_target_partially_hidden=is_target_partially_hidden, is_target_visible=is_target_visible, 
        is_reappearing=jnp.where(True,False,False), red_sensor=mo.red_sensor, green_sensor=mo.green_sensor, 
        collision_branch=collision_branch, last_collision_data=last_collision_data, 
        friction=mo.friction, elasticity=mo.elasticity, edgemap=mo.edgemap, cornermap=mo.cornermap, T=mo.T + jnp.int32(1),
        stopped_early=stopped_early
    )
