from genjax import ChoiceMapBuilder as C
from genjax import Mask
import jax.numpy as jnp
import numpy as np
import jax
from .inference_utils import extract_disjoint_rectangles
from jtap.model import *
from jtap.utils import d2r, ChexModelInput

def make_init_chm_rg(mi, init_obs_):

    init_obs = np.flip(init_obs_, axis = 1).T
    scale = mi.image_discretization
    barrier_rects = extract_disjoint_rectangles(init_obs == 3, scale = scale)
    occ_rects = extract_disjoint_rectangles(init_obs == 1, scale = scale)
    red_rect = extract_disjoint_rectangles(init_obs == 4, scale = scale)
    green_rect  = extract_disjoint_rectangles(init_obs == 5, scale = scale)

    num_barriers = jnp.int32(len(barrier_rects))
    num_occluders = jnp.int32(len(occ_rects))

    chm_dict = {'num_occs': num_occluders,
                'num_barriers': num_barriers}

    chm_dict['red_x'] = jnp.float32(red_rect[0][0])
    chm_dict['red_y'] = jnp.float32(red_rect[0][1])
    chm_dict['red_size_x'] = jnp.float32(red_rect[0][2])
    chm_dict['red_size_y'] = jnp.float32(red_rect[0][3])

    chm_dict['green_x'] = jnp.float32(green_rect[0][0])
    chm_dict['green_y'] = jnp.float32(green_rect[0][1])
    chm_dict['green_size_x'] = jnp.float32(green_rect[0][2])
    chm_dict['green_size_y'] = jnp.float32(green_rect[0][3])

    init_chm = C.d(chm_dict)

    barrier_chm = jax.vmap(lambda x, y, sx, sy: 
        C["barriers", :, "barrier_x"].set(x)
        .at["barriers", :, "barrier_y"].set(y)
        .at["barriers", :, "barrier_size_x"].set(sx)
        .at["barriers", :, "barrier_size_y"].set(sy)
    )( 
        jnp.array([float(barrier_rects[i][0]) if i < num_barriers else 0.0 for i in range(mi.max_num_barriers)]), 
        jnp.array([float(barrier_rects[i][1]) if i < num_barriers else 0.0 for i in range(mi.max_num_barriers)]),
        jnp.array([float(barrier_rects[i][2]) if i < num_barriers else 0.0 for i in range(mi.max_num_barriers)]),
        jnp.array([float(barrier_rects[i][3]) if i < num_barriers else 0.0 for i in range(mi.max_num_barriers)])
    )

    occ_chm = jax.vmap(lambda x, y, sx, sy: 
        C["occluders", :, "occ_x"].set(x)
        .at["occluders", :, "occ_y"].set(y)
        .at["occluders", :, "occ_size_x"].set(sx)
        .at["occluders", :, "occ_size_y"].set(sy)
    )( 
        jnp.array([float(occ_rects[i][0]) if i < num_occluders else 0.0 for i in range(mi.max_num_occ)]), 
        jnp.array([float(occ_rects[i][1]) if i < num_occluders else 0.0 for i in range(mi.max_num_occ)]),
        jnp.array([float(occ_rects[i][2]) if i < num_occluders else 0.0 for i in range(mi.max_num_occ)]),
        jnp.array([float(occ_rects[i][3]) if i < num_occluders else 0.0 for i in range(mi.max_num_occ)])
    )
    
    return C.d({
        'init_obs': jnp.array(init_obs_),
        'init' : init_chm.merge(barrier_chm).merge(occ_chm)

    })

def make_init_proposal_args_rg(mi, init_obs_):

    init_obs = np.flip(init_obs_, axis = 1).T
    scale = mi.image_discretization
    barrier_rects = extract_disjoint_rectangles(init_obs == 3, scale = scale)
    occ_rects = extract_disjoint_rectangles(init_obs == 1, scale = scale)
    num_barriers = len(barrier_rects)
    num_occ = len(occ_rects)
    fixed_barrier_rects = [
        barrier_rects[i] if i < num_barriers else (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0))
        for i in range(mi.max_num_barriers)
    ]
    fixed_occ_rects = [
        occ_rects[i] if i < num_occ else (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0))
        for i in range(mi.max_num_occ)
    ]
    def make_masked_stack(vals, check):
        return Mask(flag = check, value = tuple(vals))
    make_masked_stack_vmap = jax.vmap(make_masked_stack)
    masked_barriers = make_masked_stack_vmap(jnp.array(fixed_barrier_rects), jnp.arange(mi.max_num_barriers) < num_barriers)
    masked_occluders = make_masked_stack_vmap(jnp.array(fixed_occ_rects), jnp.arange(mi.max_num_occ) < num_occ)
    return ((mi, masked_occluders, masked_barriers, jnp.array(init_obs_)),)


# default mi template
mi_rg_ = ChexModelInput(
    scene_dim=jnp.array([20.0,20.0]),
    σ_pos=jnp.float32(0.0005),
    reappearing_σ_pos=jnp.float32(1.0),
    σ_size=jnp.float32(0.0005),
    σ_speed=jnp.float32(0.1),
    reappearing_σ_speed=jnp.float32(0.5),
    σ_NOCOL_dir=jnp.float32(175.0),
    σ_COL_dir=jnp.float32(100.0),
    reappearing_dir_σ=jnp.float32(5.0),
    σ_friction=jnp.float32(0.000005),
    σ_elasticity=jnp.float32(0.0005),
    σ_barrier=jnp.float32(0.0005),
    σ_barrier_size=jnp.float32(0.0005),
    σ_occ=jnp.float32(0.0005),
    σ_occ_size=jnp.float32(0.0005),
    σ_sensor_size=jnp.float32(0.0005),
    σ_sensor=jnp.float32(0.0005),
    flip_prob=jnp.float32(0.47),
    filter_size=jnp.int32(3),
    σ_pix_blur=jnp.float32(0.1),
    max_speed=jnp.float32(1.0),
    max_friction=jnp.float32(0.1),
    min_elasticity=jnp.float32(0.8),
    max_barrier_size=jnp.float32(20.0),
    max_num_barriers=jnp.int32(10),
    max_occ_size=jnp.float32(20.0),
    max_num_occ=jnp.int32(5),
    max_sensor_size=jnp.float32(20.0),
    image_discretization=jnp.float32(0.1),
    T=jnp.int32(80),
    num_x_grid=jnp.int32(11),
    num_y_grid=jnp.int32(11),
    grid_size_x=jnp.float32(1.0),
    grid_size_y=jnp.float32(1.0),
    shape_idxs=jnp.arange(2, dtype=jnp.int32),
    size_lims=jnp.array([0.3, 1.3], dtype=jnp.float32),
    σ_pos_sim=jnp.float32(0.0005),
    σ_size_sim=jnp.float32(0.0005),
    σ_speed_sim=jnp.float32(0.001),
    σ_NOCOL_dir_sim=jnp.float32(d2r(1)),
    σ_COL_dir_sim=jnp.float32(d2r(2)),
    max_num_col_iters = jnp.float32(10)
)
