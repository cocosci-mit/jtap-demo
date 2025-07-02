import jax
import genjax
import jax.numpy as jnp
from genjax import gen
from jtap.distributions import uniformposition2d
from jtap.utils import ModelOutput
from .scene_geometry import get_edges_from_scene, get_corners_from_scene

def is_circle_intersecting_box(box_x, box_y, box_w, box_h, circle_x, circle_y, radius):
    closest_x = jnp.clip(circle_x, box_x, box_x + box_w)
    closest_y = jnp.clip(circle_y, box_y, box_y + box_h)
    
    dist_sq = (closest_x - circle_x) ** 2 + (closest_y - circle_y) ** 2
    
    return dist_sq <= radius ** 2

def is_target_intersecting_rectangle_inner(masked_rectangle, x, y, size):
    occ_x, occ_y, occ_size_x, occ_size_y = masked_rectangle.value
    r = size/2
    return jnp.logical_and(
        masked_rectangle.flag,
        is_circle_intersecting_box(occ_x, occ_y, occ_size_x, occ_size_y, x+r, y+r, r)
    )

@jax.jit
def is_target_intersecting_rectangle(x, y, size, masked_rectangles):
    return jnp.any(jax.vmap(is_target_intersecting_rectangle_inner, in_axes = (0,None,None,None))(masked_rectangles, x, y, size))


@jax.jit
def is_target_fully_hidden_inner(masked_occluder, x, y, size):
    occ_x, occ_y, occ_size_x, occ_size_y = masked_occluder.value
    return jnp.logical_and(jnp.all(
        jnp.array([
            x + size <= occ_x + occ_size_x, 
            x >= occ_x,
            y + size <= occ_y + occ_size_y, 
            y >= occ_y
        ])
    ), masked_occluder.flag)

@jax.jit
def is_target_fully_hidden(x, y, size, masked_occluders):
    return jnp.any(jax.vmap(is_target_fully_hidden_inner, in_axes = (0,None,None,None))(masked_occluders, x, y, size))

# sample single occluder
@gen
def sample_occluder(occ_params):
    occ_size_x = genjax.uniform(jnp.float32(0.),occ_params[0]) @ "occ_size_x"
    occ_size_y = genjax.uniform(jnp.float32(0.),occ_params[0]) @ "occ_size_y"
    occ_x = genjax.uniform(jnp.float32(0.),occ_params[1][0]) @ "occ_x"
    occ_y = genjax.uniform(jnp.float32(0.), occ_params[1][1]) @ "occ_y"
    return (occ_x, occ_y, occ_size_x, occ_size_y)

sample_occluder_masked = genjax.mask(sample_occluder)
sample_occluder_masked_map = genjax.vmap(in_axes=(0,None))(sample_occluder_masked)

# sample single barrier
@gen
def sample_barrier(barrier_params):
    barrier_size_x = genjax.uniform(jnp.float32(0.),barrier_params[0]) @ "barrier_size_x"
    barrier_size_y = genjax.uniform(jnp.float32(0.),barrier_params[0]) @ "barrier_size_y"
    barrier_x = genjax.uniform(jnp.float32(0.),barrier_params[1][0]) @ "barrier_x"
    barrier_y = genjax.uniform(jnp.float32(0.), barrier_params[1][1]) @ "barrier_y"
    return (barrier_x, barrier_y, barrier_size_x, barrier_size_y)

sample_barrier_masked = genjax.mask(sample_barrier)
sample_barrier_masked_map = genjax.vmap(in_axes=(0, None))(sample_barrier_masked)

@gen
def init_model(mi):

    num_occs = genjax.categorical(mi.occ_num_probs) @ "num_occs"
    num_barriers = genjax.categorical(mi.barrier_num_probs) @ "num_barriers"

    masked_occluders = sample_occluder_masked_map(
        jnp.arange(1, mi.occ_num_probs.shape[0]) <= num_occs,
        (mi.max_occ_size, mi.scene_dim)
    ) @ "occluders"

    masked_barriers = sample_barrier_masked_map(
        jnp.arange(1, mi.barrier_num_probs.shape[0]) <= num_barriers,
        (mi.max_barrier_size, mi.scene_dim)
    ) @ "barriers"

    # Square length OR Circle diameter
    size = genjax.uniform(*mi.size_lims) @ 'size'

    # 0 is square, 1 is circle;
    # shape = genjax.uniform(unicat(mi.shape_idxs)) @ 'shape' 
    x, y = uniformposition2d(jnp.float32(0.), mi.scene_dim[0] - size, jnp.float32(0.), mi.scene_dim[1] - size) @ "xy" # joint sampling of x and y
    speed = genjax.uniform(jnp.float32(0.), mi.max_speed) @ "speed"
    dir = genjax.uniform(-jnp.pi, jnp.pi) @ "dir"
    friction = genjax.uniform(jnp.float32(0.), mi.max_friction) @ "friction"
    elasticity = genjax.uniform(mi.min_elasticity, jnp.float32(1.)) @ "elasticity"

    edgemap = get_edges_from_scene(mi.scene_dim, mi.empty_edgemap, masked_barriers)
    cornermap = get_corners_from_scene(mi.scene_dim, mi.empty_cornermap, masked_barriers)
    
    # sample red and green regions
    red_size_x = genjax.uniform(jnp.float32(0.), mi.max_sensor_size) @ "red_size_x"
    red_size_y = genjax.uniform(jnp.float32(0.), mi.max_sensor_size) @ "red_size_y"
    red_x = genjax.uniform(jnp.float32(0.), mi.scene_dim[0]) @ "red_x"
    red_y = genjax.uniform(jnp.float32(0.), mi.scene_dim[1]) @ "red_y"
    red_sensor = jnp.array([red_x, red_y, red_size_x, red_size_y])
   
    green_size_x = genjax.uniform(jnp.float32(0.), mi.max_sensor_size) @ "green_size_x"
    green_size_y = genjax.uniform(jnp.float32(0.), mi.max_sensor_size) @ "green_size_y"
    green_x = genjax.uniform(jnp.float32(0.), mi.scene_dim[0]) @ "green_x"
    green_y = genjax.uniform(jnp.float32(0.), mi.scene_dim[1]) @ "green_y"
    green_sensor = jnp.array([green_x, green_y, green_size_x, green_size_y])

    # bug workaround for not being able to return non-traced values for importance
    collision_branch = jax.lax.select(True, 4., 4.).astype(jnp.float32)
    is_reappearing = jax.lax.select(True, False, False).astype(jnp.bool)
    shape = jax.lax.select(True, 1, 1).astype(jnp.int32)
    # timestep, x, y, speed, dir
    last_collision_data = jnp.array([0,x,y,speed,dir])

    # check visibility condition
    is_target_hidden = is_target_fully_hidden(x, y, size, masked_occluders)
    is_target_visible = jnp.logical_not(is_target_intersecting_rectangle(x, y, size, masked_occluders))
    is_target_partially_hidden = jnp.logical_not(jnp.logical_or(is_target_hidden, is_target_visible))

    return ModelOutput(
        shape, size, x, y, speed, dir, num_occs, num_barriers,
        masked_occluders, masked_barriers, is_target_hidden,
        is_target_partially_hidden, is_target_visible, is_reappearing,
        red_sensor, green_sensor, collision_branch, last_collision_data,
        elasticity, friction, edgemap, cornermap, jax.lax.select(True, 0, 0).astype(jnp.int32),
        jax.lax.select(True, False, False).astype(jnp.bool)
    )
