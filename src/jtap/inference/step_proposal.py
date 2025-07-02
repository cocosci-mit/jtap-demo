import jax
import jax.numpy as jnp
from genjax import gen, ChoiceMap
from genjax import ChoiceMapBuilder as C
from jtap.distributions import *
from jtap.model import *
from jtap.model import maybe_resolve_collision

@gen
def step_proposal_propose_occlusion(mi, col_next_speed, col_next_dir, col_next_x, col_next_y, col_branch):

    dir_noise = jnp.where(col_branch == jnp.float32(4), mi.σ_NOCOL_dir_occ, mi.σ_COL_dir_occ)

    proposed_speed = genjax.truncated_normal(col_next_speed, mi.σ_speed_occ, jnp.float32(0), mi.max_speed) @ "speed"
    proposed_dir = circular_normal(col_next_dir, dir_noise)@ "dir"
    return proposed_speed, proposed_dir, col_next_x, col_next_y

@gen
def step_proposal_propose_vel_and_pos(mi, speed_mean, dir_mean, proposed_x, proposed_y, t):
    proposed_dir_noise = jax.lax.select(
        jnp.less_equal(t, jnp.int32(2)),
        mi.σ_NOCOL_dir_initprop,
        mi.σ_NOCOL_dir_stepprop
    )

    proposed_speed_noise = jax.lax.select(
        jnp.less_equal(t, jnp.int32(2)),
        mi.σ_speed_initprop,
        mi.σ_speed_stepprop
    )
    proposed_speed = genjax.truncated_normal(speed_mean, proposed_speed_noise, jnp.float32(0), mi.max_speed) @ "speed"
    proposed_dir = circular_normal(dir_mean, proposed_dir_noise)@ "dir"
    return proposed_speed, proposed_dir, proposed_x, proposed_y

@gen
def step_proposal_propose_collision(mi, col_next_speed, col_next_dir, proposed_x, proposed_y):

    proposed_speed = genjax.truncated_normal(col_next_speed, mi.σ_speed_stepprop, jnp.float32(0), mi.max_speed) @ "speed"
    proposed_dir = circular_normal(col_next_dir, mi.σ_COL_dir_prop)@ "dir"
    return proposed_speed, proposed_dir, proposed_x, proposed_y

step_proposal_switch = genjax.switch(
    step_proposal_propose_vel_and_pos, 
    step_proposal_propose_occlusion,
    step_proposal_propose_collision
)

@gen
def step_proposal(mi, is_fully_hidden, mo, proposed_x, proposed_y, t):

    proposed_x = jnp.clip(proposed_x, jnp.float32(0), mi.scene_dim[0] - mo.size)
    proposed_y = jnp.clip(proposed_y, jnp.float32(0), mi.scene_dim[1] - mo.size)

    masked_barriers = mo.masked_barriers
    masked_occluders = mo.masked_occluders

    last_collision_T, last_collision_x, last_collision_y = mo.last_collision_data[:3]

    # velocity gradient since last collision
    vx, vy = ((proposed_x - last_collision_x)/(t - last_collision_T), (proposed_y - last_collision_y)/(t - last_collision_T))
    
    speed_mean = jnp.sqrt(jnp.square(vx) + jnp.square(vy))
    speed_mean = jnp.clip(speed_mean, jnp.float32(0) + jnp.finfo(float).eps, mi.max_speed - jnp.finfo(float).eps)

    dir_mean = jnp.arctan2(vy, vx)

    prev_vx, prev_vy = mo.speed * jnp.cos(mo.dir), mo.speed * jnp.sin(mo.dir)
    col_next_x, col_next_y, col_next_vx, col_next_vy, col_branch = maybe_resolve_collision(mo.size, mo.x, mo.y, prev_vx, prev_vy, mo.edgemap, mo.cornermap, mo.speed)[0][-5:]
    col_next_speed = jnp.sqrt(jnp.square(col_next_vx) + jnp.square(col_next_vy))
    col_next_dir = jnp.arctan2(col_next_vy, col_next_vx)

    # clip x and y positions to range
    col_next_x = jnp.clip(col_next_x, jnp.float32(0), mi.scene_dim[0] - mo.size)
    col_next_y = jnp.clip(col_next_y, jnp.float32(0), mi.scene_dim[1] - mo.size)

    use_collision_proposal = jnp.logical_and(
        jnp.greater(t,jnp.int32(1)),
        jnp.not_equal(col_branch, jnp.float32(4.))
    )
    
    use_occlusion_proposal = is_fully_hidden

    switch_branch = jnp.where(
        use_occlusion_proposal,
        jnp.int32(1),
        jnp.where(
            use_collision_proposal,
            jnp.int32(2),
            jnp.int32(0)
        )
    )


    proposed_speed, proposed_dir, step_prop_x, step_prop_y = step_proposal_switch(switch_branch, 
        (mi, speed_mean, dir_mean, proposed_x, proposed_y, t),
        (mi, col_next_speed, col_next_dir, col_next_x, col_next_y, col_branch),
        (mi, col_next_speed, col_next_dir, proposed_x, proposed_y)
    ) @ 'prop_switch' 

    return {
        "t" : t,
        "proposal_branch": switch_branch,
        "speed_mean": speed_mean,
        "dir_mean": dir_mean,
        "vx": vx,
        "vy": vy,
        "last_collision_T": last_collision_T,
        "last_collision_x": last_collision_x,
        "last_collision_y": last_collision_y,
        "proposed_x": proposed_x, # what comes from GRID inference
        "proposed_y": proposed_y,
        "step_prop_x": step_prop_x, # what comes from internal proposal
        "step_prop_y": step_prop_y,
        "col_branch": col_branch,
        "proposed_speed": proposed_speed,
        "proposed_dir": proposed_dir,
        "col_values" : {
            "col_next_dir" : col_next_dir,
            "col_next_speed" : col_next_speed,
            "col_next_x": col_next_x,
            "col_next_y": col_next_y,
            "mo_size": mo.size,
            "mo_shape": mo.shape,
            "mo_x": mo.x,
            "mo_y": mo.y,
            "prev_vx": prev_vx,
            "prev_vy": prev_vy,
            "mo_edgemap": mo.edgemap,
            "mo_cornermap": mo.cornermap,
            "mo_speed": mo.speed,
            "mo_dir": mo.dir
        },
        "use_occlusion_proposal": use_occlusion_proposal,
    }



def step_choicemap_translator(cm_data_driven_proposal, cm_grid_inference_proposal, cm_obs):

    return C.d(
            {
                'step' : cm_data_driven_proposal('prop_switch').merge(cm_grid_inference_proposal),
                'obs' : cm_obs['obs']
            }
        )