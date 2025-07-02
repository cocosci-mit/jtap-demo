import jax
import os
# Create .jaxcache directory if it doesn't exist
jax_cache_dir = os.path.expanduser("~/.jaxcache")
if not os.path.exists(jax_cache_dir):
    os.makedirs(jax_cache_dir)
jax.config.update("jax_compilation_cache_dir", jax_cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.experimental.compilation_cache.compilation_cache.set_cache_dir(jax_cache_dir)
import jax.numpy as jnp
import numpy as np
import time
import random
import os
from tqdm import tqdm
from termcolor import colored
import genjax
from genjax import gen
from genjax import ChoiceMapBuilder as C
from jtap.all import *
import jax.tree_util as jtu
import numpy as np
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib import font_manager

# Path to the font file

font_path = os.path.join(get_fonts_dir(), "DMSans-Regular.ttf")
dm_sans_prop = font_manager.FontProperties(fname=font_path)
font_manager.fontManager.addfont(font_path)
font_manager.fontManager.addfont(os.path.join(get_fonts_dir(), "DMSans-Bold.ttf"))

print(colored("Imports successful", "green"))

def list_folders_in_path(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

cogsci_folder = os.path.join(get_assets_dir(), 'cogsci2025')
global_trial_names = sorted([x for x in list_folders_in_path(cogsci_folder) if x.startswith('E')], key=lambda x: int(x[1:]))
human_data_pkl = os.path.join(cogsci_folder, 'cogsci2025_human_data.pkl')

# This is specific to the CogSci 2025 stimuli
trials_with_occlusion = [
    'E1', 'E3', 'E5', 'E7', 'E9', 'E11', 'E13', 'E17', 'E19', 'E21', 'E23', 'E25', 'E27', 'E29',
    'E31', 'E32', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39', 'E40', 'E41', 'E42', 'E43',
    'E44', 'E45', 'E46', 'E47', 'E48', 'E49', 'E50'
]

skip_t = 4
trial_frame_counts = []
observations = []
observations_np = []
padded_observations = []
ALL_rgb_videos_full_frames = {}
trial_name_to_idx = {}
for i, trial_name in tqdm(enumerate(global_trial_names), desc = "Preprocessing stimuli"):
    rgb_video, onehot_video_obs, onehot_video_obs_np = load_stimuli(trial_name, cogsci_folder, skip_t = skip_t)
    trial_frame_counts.append(onehot_video_obs.shape[0])
    observations.append(onehot_video_obs)
    observations_np.append(onehot_video_obs_np)
    trial_name_to_idx[trial_name] = i
    ALL_rgb_videos_full_frames[trial_name] = rgb_video
max_num_frames = max(trial_frame_counts)
num_trials = len(trial_frame_counts)

for x in observations:
    padded_observations.append(pad_obs_with_last_frame(x, max_num_frames - x.shape[0]))

session_df, trial_df, keystate_df, position_data, occlusion_durations, occlusion_frames,\
    HUMAN_stacked_key_presses, HUMAN_stacked_key_dist, HUMAN_scores, \
    HUMAN_stacked_scores, HUMAN_stacked_key_SWITCHES = get_human_data(human_data_pkl, skip_t, global_trial_names)



init_proposer= jax.vmap(init_proposal.propose, in_axes = (0,None))
init_choicemap_merger = jax.vmap(init_choicemap_translator, in_axes = (0, None))
step_proposer= jax.vmap(step_proposal.propose, in_axes = (0,(None,None,0,0,0,None)))
step_choicemap_merger = jax.vmap(step_choicemap_translator, in_axes = (0,0,None))
initializer = jax.vmap(full_init_model.importance, in_axes = (0,0,None))
stepper = jax.vmap(full_step_model.importance, in_axes = (0,0,(0,None,None)))
simulator = jax.vmap(stepper_model.simulate, in_axes = (0,(0,None,None)))
idx_resampler = jax.vmap(genjax.categorical.simulate, in_axes=(0, None))
mo_resampler = jax.vmap(lambda idx, mos:jtu.tree_map(lambda v : v[idx], mos), in_axes = (0, None))

def find_valid_positions_bool(pos_chm, size, masked_barriers):
    return jnp.logical_not(jax.vmap(is_target_intersecting_rectangle, in_axes=(0,0,None,None))(pos_chm['x'], pos_chm['y'], size, masked_barriers))

likelihood_grid_evaluator = jax.vmap(
    likelihood_model.logpdf,
    in_axes = (None,(
        None,None,None,None,0,0,None,None,None,None
    ),None)
)

@gen
# NOTE: C2F with my strategy is just to add up the weights
def sample_grid(grid_scores, pos_cells):
    sampled_grid_idx = genjax.categorical(grid_scores) @ 'sampled_grid_index'
    x_cell = pos_cells[sampled_grid_idx, 0:2]
    y_cell = pos_cells[sampled_grid_idx, 2:4]
    x = genjax.uniform(x_cell[0], x_cell[1]) @ 'x'
    y = genjax.uniform(y_cell[0], y_cell[1]) @ 'y'
    return x,y,sampled_grid_idx,x_cell,y_cell

grid_proposer = jax.vmap(sample_grid.propose, in_axes=(0,None))

# NOTE: DISABLED TAIL FOR NOW
def make_pos_chm_grid(mi, x_center, y_center, x_size, y_size):
    # NOTE: Not using multivmaps, but a flattened vmap (more efficient for a 2D grid)
    # preferably have odd numbered grids
    # make grid with num_x by num_y
    num_x_grid = mi.num_x_grid_arr.shape[0]
    num_y_grid = mi.num_y_grid_arr.shape[0]
    min_x, max_x = x_center - x_size/2, x_center + x_size/2
    min_y, max_y = y_center - y_size/2, y_center + y_size/2

    # Clip to scene bounds
    min_x, max_x = jnp.clip(min_x, jnp.float32(0.0), mi.scene_dim[0] - jnp.float32(1.0)), jnp.clip(max_x, jnp.float32(0.0), mi.scene_dim[0] - jnp.float32(1.0))
    min_y, max_y = jnp.clip(min_y, jnp.float32(0.0), mi.scene_dim[1] - jnp.float32(1.0)), jnp.clip(max_y, jnp.float32(0.0), mi.scene_dim[1] - jnp.float32(1.0))
    # create deterministic grid points
    x_grid = jnp.linspace(min_x, max_x, num_x_grid)
    y_grid = jnp.linspace(min_y, max_y, num_y_grid)
    # add in tail points
    # x_grid = jnp.concat([jnp.array([(min_x)/2]),x_grid,jnp.array([(max_x + mi.scene_dim[0])/2])], axis = 0)
    # y_grid = jnp.concat([jnp.array([(min_y)/2]),y_grid,jnp.array([(max_y + mi.scene_dim[1])/2])], axis = 0)
    
    # make meshgrid and choicemap grid
    x_meshgrid, y_meshgrid = jnp.meshgrid(x_grid, y_grid, indexing = 'ij')
    x_meshgrid_flat, y_meshgrid_flat = x_meshgrid.flatten(), y_meshgrid.flatten()
    pos_chm_grid = jax.vmap(lambda x,y: C['x'].set(x).at['y'].set(y))(x_meshgrid_flat, y_meshgrid_flat)
    # make cells/intervals and modify tail bounds
    x_div = x_grid[2] - x_grid[1]
    y_div = y_grid[2] - y_grid[1]
    pos_cells = jnp.stack([x_meshgrid_flat - x_div/2, x_meshgrid_flat + x_div/2, y_meshgrid_flat - y_div/2, y_meshgrid_flat + y_div/2], axis = 1)
    # lower_tail_x_upper = jnp.max(jnp.array([x_div/2, min_x - x_div/2]))
    # lower_tail_y_upper = jnp.max(jnp.array([y_div/2, min_y - y_div/2]))
    # upper_tail_x_lower = jnp.min(jnp.array([max_x + x_div/2, mi.scene_dim[0]-x_div/2]))
    # upper_tail_y_lower = jnp.min(jnp.array([max_y + y_div/2, mi.scene_dim[1]-y_div/2]))
    # pos_cells = pos_cells.at[::num_y_grid+2,2:4].set(jnp.array([0, lower_tail_y_upper]))
    # pos_cells = pos_cells.at[num_y_grid+1::num_y_grid+2,2:4].set(jnp.array([upper_tail_y_lower, mi.scene_dim[1]]))
    # pos_cells = pos_cells.at[:num_x_grid+2,0:2].set(jnp.array([0, lower_tail_x_upper]))
    # pos_cells = pos_cells.at[-num_x_grid-2:,0:2].set(jnp.array([upper_tail_x_lower, mi.scene_dim[0]]))

    return pos_chm_grid, pos_cells

# NOTE: this depends on the need of what tracking data is needed (inference)
def get_JTAP_data(mos):
    return {
        'tracking' :
        {
            'x' : mos.x,
            'y' : mos.y,
            'size' : mos.size,
            'dir' : mos.dir,
            'speed' : mos.speed,
        }
    }

# @partial(jax.jit, static_argnames=["max_inference_steps", "num_particles"])
def sequential_monte_carlo_rg(initial_key, mi, ESS_proportion, init_proposal_args, 
            init_inference_chm, onehot_video_obs, max_inference_steps, num_particles):

    def resample_fn(weights, step_mos, resample_key):
        idxs = idx_resampler(jax.random.split(resample_key, num_particles), (weights,)).get_retval()
        mos_resampled = mo_resampler(idxs, step_mos)
        return mos_resampled, jnp.zeros_like(weights)

    def prediction(carry, _):
        key, sim_mos, coded_rg = carry
        next_key, sim_key = jax.random.split(key, 2)
        sim_keys = jax.random.split(sim_key, num_particles)
        next_sim_mos = simulator(sim_keys, (sim_mos, mi, False)).get_retval()
        stopped_early = jnp.any(next_sim_mos.stopped_early)
        coded_rg = jnp.where(coded_rg == jnp.int8(0), determine_sensor_output_from_mo_vmap(next_sim_mos), coded_rg)
        prediction_data = {'x': next_sim_mos.x, 'y' : next_sim_mos.y, 'rg' : coded_rg}
        return (next_key, next_sim_mos, coded_rg), (prediction_data, stopped_early)

    def particle_filter_step_rg(carry, t):
        key, mos, weights, prev_num_obj_pixels = carry

        SINGLE_MO = slice_pt(mos,0)

        # Split main keys
        next_key, sim_key, stepper_key, proposal_key, resample_key = jax.random.split(key, 5)
        data_driven_proposal_key, grid_proposal_key = jax.random.split(proposal_key, 2)
        grid_sampling_key, _ = jax.random.split(grid_proposal_key, 2)
        # vmap keys
        stepper_keys = jax.random.split(stepper_key, num_particles)
        data_driven_proposal_keys = jax.random.split(data_driven_proposal_key, num_particles)
        grid_sampling_keys = jax.random.split(grid_sampling_key, num_particles)
        # grid_scoring_keys = jax.random.split(grid_scoring_key, num_particles)

        # get obs choicemap
        step_obs_chm = C.d({
                "obs": onehot_video_obs[t],
            })
        #########################################
        # GRID INFERENCE (STEP)
        #########################################
        _, x_grid_center, y_grid_center, obs_is_fully_hidden_, num_obj_pixels = data_driven_size_and_position(onehot_video_obs[t], mi.image_discretization)

        # see if object is reappearing
        data_driven_is_reappearing = jnp.where(
            num_obj_pixels > 0.85 * 80,
            jnp.bool_(False),
            jnp.where(
                jnp.greater(num_obj_pixels, prev_num_obj_pixels*1.15),
                jnp.bool_(True),
                jnp.bool_(False)
            )
        )

        # see if object is dis-appearing with low_pixel count
        data_driven_is_disappearing_low_pix_count = jnp.logical_and(
            num_obj_pixels < 0.2 * 80,
            jnp.logical_not(data_driven_is_reappearing)
        )

        # make object FULLY HIDDEN If DISAPPERING with low pixel count
        obs_is_fully_hidden = jnp.logical_or(obs_is_fully_hidden_, data_driven_is_disappearing_low_pix_count)

        # data driven GRID SIZE
        grid_size_x, grid_size_y = jax.lax.cond(
            num_obj_pixels > 0.8 * 80, # NOTE: this is wrt 200x200 images
            lambda: (mi.grid_size_x, mi.grid_size_y),
            lambda: jax.lax.cond(
                num_obj_pixels > 0.5 * 80,
                lambda: (jnp.float32(1.0), jnp.float32(1.0)),
                lambda: (jnp.float32(1.7), jnp.float32(1.7))
            )
        )
        # step model args
        inference_mode_bool = True # jnp.where(obs_is_fully_hidden, False, True)
        step_model_args = (mos, mi, inference_mode_bool)

        #TODO: GRIDS must NOT allow positions stuck inside barriers
        pos_chm_grid, pos_cells = make_pos_chm_grid(mi, x_grid_center, y_grid_center, grid_size_x, grid_size_y)
        valid_position_bools = find_valid_positions_bool(pos_chm_grid, jnp.float32(1.0), SINGLE_MO.masked_barriers) 

        # EVALUATE GRID SCORES
        total_grid_size = pos_cells.shape[0]
        observed_image = step_obs_chm['obs']
        grid_render_args = (
            mi.pix_x, mi.pix_y, SINGLE_MO.shape, jnp.float32(1.0),
            pos_chm_grid['x'], pos_chm_grid['y'], SINGLE_MO.masked_barriers, SINGLE_MO.masked_occluders,
            SINGLE_MO.red_sensor, SINGLE_MO.green_sensor
        )

        grid_scores = likelihood_grid_evaluator(observed_image,grid_render_args, mi.flip_prob)
        grid_scores = jnp.where(valid_position_bools, grid_scores, jnp.array(-jnp.inf))

        # SAMPLE GRID and CONTINUOUS SAMPLE
        _, grid_weights, grid_retvals = grid_proposer(grid_sampling_keys, (grid_scores, pos_cells))
        (proposed_xs, proposed_ys, sampled_idxs, x_cells, y_cells) = grid_retvals

        
        grid_details = {
            'proposed_xs' : proposed_xs,
            'proposed_ys' : proposed_ys,
            'sampled_idxs' : sampled_idxs,
            'x_cells' : x_cells,
            'y_cells' : y_cells,
            'grid_size_x' : grid_size_x,
            'grid_size_y' : grid_size_y,
            'grid_x' : pos_chm_grid['x'],
            'grid_y' : pos_chm_grid['y'],
            'grid_scores' : grid_scores,
        } 

        # Propose new step particle extensions
        step_proposal_args = (mi, obs_is_fully_hidden, mos, proposed_xs, proposed_ys, t)
        step_proposed_choices, step_prop_weights, step_prop_retval = step_proposer(data_driven_proposal_keys, step_proposal_args)
        # Merge choicemap

        step_prop_xs, step_prop_ys = step_prop_retval['step_prop_x'], step_prop_retval['step_prop_y']

        # SWITCH CHM for Occlusion
        position_choices = jax.vmap(lambda x,y: C['x'].set(x).at['y'].set(y))(step_prop_xs, step_prop_ys)

        step_inference_chm_merged = step_choicemap_merger(step_proposed_choices, position_choices, step_obs_chm)
        
        # Update particles
        step_particles, incremental_weights = stepper(stepper_keys, step_inference_chm_merged, step_model_args)
        step_mos = step_particles.get_retval()

        grid_weights = jnp.where(obs_is_fully_hidden, jnp.zeros_like(grid_weights), grid_weights)

        weight_composition = {
            'step_prop_weights' : step_prop_weights,
            'grid_weights' : grid_weights,
            'incremental_weights' : incremental_weights,
            'prev_weights' : weights,
            'inference_mode_bool' : inference_mode_bool
        }

        # Update weights
        weights = weights + incremental_weights - step_prop_weights - grid_weights
        
        # Resample if ESS is low
        ESS = effective_sample_size(weights)
        
        step_mos, weights = jax.lax.cond(
            ESS < ESS_threshold,
            resample_fn,
            lambda *_ : (step_mos, weights),
            weights, step_mos, resample_key
        )


        _, (prediction_data, pred_stopped_early) = jax.lax.scan(
            prediction,
            (sim_key, step_mos, determine_sensor_output_from_mo_vmap(step_mos)),
            jnp.arange(max_prediction_steps)
        )

        stopped_early = jnp.logical_or(jnp.any(step_mos.stopped_early), jnp.any(pred_stopped_early))

        JTAP_data_step = get_JTAP_data(step_mos)
        JTAP_data_step['prediction'] = prediction_data # joint tracking and prediction
        JTAP_data_step['weights'] = weights
        JTAP_data_step['t'] = t
        JTAP_data_step['resampled'] = ESS < ESS_threshold
        JTAP_data_step['ESS'] = ESS
        JTAP_data_step['is_target_hidden'] = step_mos.is_target_hidden
        JTAP_data_step['is_target_partially_hidden'] = step_mos.is_target_partially_hidden
        JTAP_data_step['obs_is_fully_hidden'] = obs_is_fully_hidden
        JTAP_data_step['stopped_early'] = stopped_early
        
        return (next_key, step_mos, weights, num_obj_pixels), (JTAP_data_step, step_prop_retval, (weight_composition, grid_details))

    # Prepare initial carry state
    max_prediction_steps = max_inference_steps + 5 # fix to max num inference steps
    ESS_threshold = ESS_proportion * num_particles
    key, subkey, smc_key, sim_key = jax.random.split(initial_key,4)
    initializer_keys = jax.random.split(key, num_particles)
    proposal_keys = jax.random.split(subkey, num_particles)

    # only using this to get obs_is_fully_hidden
    *_, obs_is_fully_hidden, num_obj_pixels = data_driven_size_and_position(onehot_video_obs[0], mi.image_discretization)

    # Initial proposal
    init_proposed_choices, init_prop_weights, init_prop_retval = init_proposer(proposal_keys, init_proposal_args)    
    init_inference_chm_merged = init_choicemap_merger(init_proposed_choices, init_inference_chm)
    model_args = (mi,)

    # return init_inference_chm_merged

    init_particles, init_weights = initializer(initializer_keys, init_inference_chm_merged, model_args)
    initial_weights = init_weights - init_prop_weights
    init_mos = init_particles.get_retval()

    sample_mo = slice_pt(init_mos,0)
    sample_obs = likelihood_model.sample(jax.random.PRNGKey(0), get_render_args(mi,sample_mo), mi.flip_prob)

    _, (init_prediction_data, pred_stopped_early) = jax.lax.scan(
            prediction,
            (sim_key, init_mos, determine_sensor_output_from_mo_vmap(init_mos)),
            jnp.arange(max_prediction_steps)
        )
    
    
    stopped_early = jnp.logical_or(jnp.any(init_mos.stopped_early), jnp.any(pred_stopped_early))

    JTAP_data_init = get_JTAP_data(init_mos)
    JTAP_data_init['prediction'] = init_prediction_data
    JTAP_data_init['weights'] = initial_weights
    JTAP_data_init['t'] = jnp.int32(0)
    JTAP_data_init['resampled'] = jnp.bool(False)
    JTAP_data_init['ESS'] = effective_sample_size(initial_weights)
    JTAP_data_init['is_target_hidden'] = init_mos.is_target_hidden
    JTAP_data_init['is_target_partially_hidden'] = init_mos.is_target_partially_hidden
    JTAP_data_init['obs_is_fully_hidden'] = obs_is_fully_hidden
    JTAP_data_init['stopped_early'] = stopped_early

    # Initial carry state
    initial_carry = (
        smc_key,
        init_mos,
        initial_weights,
        num_obj_pixels
    )
    
    # Perform scan
    _, (JTAP_data, step_prop_retvals, supp_data) = jax.lax.scan(
        particle_filter_step_rg, 
        initial_carry, 
        jnp.arange(1,max_inference_steps)  # Start from 1
    )

    JTAP_data = init_step_concat(JTAP_data_init, JTAP_data)
    JTAP_data['JTAP_params'] = {
        'image_discretization' : mi.image_discretization,
        'max_prediction_steps' : max_prediction_steps,
        'max_inference_steps' : max_inference_steps,
        'num_particles' : num_particles
    }
    # debug mode
    JTAP_data['step_prop_retvals'] = step_prop_retvals
    JTAP_data['init_prop_retval'] = init_prop_retval
    JTAP_data['supp_data'] = supp_data
    JTAP_data['sample_obs'] = sample_obs

    return JTAP_data

# First, apply vmap to vectorize
sequential_monte_carlo_rg_vmap_ = jax.vmap(sequential_monte_carlo_rg, in_axes=(0, None, None, None, None, None, None, None))

sequential_monte_carlo_rg_vmap = jax.jit(sequential_monte_carlo_rg_vmap_, static_argnames=["max_inference_steps", "num_particles"])

# TARGET MODEL PARAMS -- NORMAL SCENARIO

Model_Input = ChexModelInput(
    scene_dim=jnp.array([20.0,20.0]),
    σ_pos=jnp.float32(0.75),
    reappearing_σ_pos=jnp.float32(1.0),
    σ_size=jnp.float32(0.0005),
    σ_speed=jnp.float32(0.15),
    reappearing_σ_speed=jnp.float32(0.5),
    σ_NOCOL_dir=d2r(15),
    σ_COL_dir=d2r(15),
    reappearing_dir_σ=jnp.float32(5.0),
    σ_friction=jnp.float32(0.000005),
    σ_elasticity=jnp.float32(0.0005),
    σ_barrier=jnp.float32(0.0005),
    σ_barrier_size=jnp.float32(0.0005),
    σ_occ=jnp.float32(0.0005),
    σ_occ_size=jnp.float32(0.0005),
    σ_sensor_size=jnp.float32(0.0005),
    σ_sensor=jnp.float32(0.0005),
    flip_prob=jnp.float32(0.01),
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
    num_x_grid=jnp.int32(21),
    num_y_grid=jnp.int32(21),
    grid_size_x=jnp.float32(0.2),
    grid_size_y=jnp.float32(0.2),
    shape_idxs=jnp.arange(2, dtype=jnp.int32),
    size_lims=jnp.array([0.3, 1.3], dtype=jnp.float32),
    σ_pos_sim=jnp.float32(0.0005),
    σ_size_sim=jnp.float32(0.0005),
    σ_speed_sim=jnp.float32(0.0005),
    σ_NOCOL_dir_sim=d2r(0.8),
    σ_COL_dir_sim=d2r(1.6),
    max_num_col_iters=jnp.float32(5),
    σ_speed_occ=jnp.float32(0.0005),
    σ_NOCOL_dir_occ=d2r(0.8),
    σ_COL_dir_occ=d2r(1.6),
    σ_size_initprop=jnp.float32(0.01),
    σ_pos_initprop=jnp.float32(0.02),
    σ_speed_initprop=jnp.float32(0.1),
    σ_speed_stepprop=jnp.float32(0.04),
    σ_NOCOL_dir_initprop=d2r(0.3),
    σ_NOCOL_dir_stepprop=d2r(4.0),
    σ_COL_dir_prop=d2r(0.3)
)
# PREPARE INPUT
Model_Input.prepare_input()


model_tuned_setup = {
    'setup_name' : 'JTAP',
    'num_particles' : 25,
    'simnoise' : 1.75,
    'dir_stepprop_noise' : 3.0,
    'flip_prob' : 0.01,
    'speed_noise' : 0.0001,
}

frozen_tuned_setup = {
    'setup_name' : 'Frozen_Ablation',
    'num_particles' : 25,
    'simnoise' : 1.5,
    'dir_stepprop_noise' : 2.25,
    'flip_prob' : 0.45,
    'speed_noise' : 0.0001,
}

decaying_tuned_setup = {
    'setup_name' : 'Decaying_Ablation',
    'num_particles' : 25,
    'simnoise' : 1.5,
    'dir_stepprop_noise' : 1.5,
    'flip_prob' : 0.45,
    'speed_noise' : 0.05,
}

# setting random seeds for sampling from the builtin random lib, this DOES not apply to jax.random.PRNGKey
random.seed(0)
COGSCI_CONFIG = {}
for setup_ in [model_tuned_setup, frozen_tuned_setup, decaying_tuned_setup]:
    COGSCI_CONFIG[setup_['setup_name']] = {
        'num_particles' : setup_['num_particles'],
        'flip_prob' : setup_['flip_prob'],
        'ESS_threshold' : 0.15, # Effective Sample Size is always 15% of num_particles
        'movement_noise' : setup_['simnoise'],
        'collision_noise' : setup_['simnoise'],
        'speed_noise' : setup_['speed_noise'],
        'speed_noise_occ' : setup_['speed_noise'],
        'movement_noise_occ' : setup_['simnoise'],
        'collision_noise_occ' : setup_['simnoise'],
        'num_experiments' : 100,
        'dir_stepprop_noise' : setup_['dir_stepprop_noise'],
    }
num_setups = len(COGSCI_CONFIG)

print(colored("Executing 3 setups for all trials, each tuned for JTAP, Frozen, and Decaying respectively", "cyan"))
print(colored(f"Total number of seeded runs per setup: {list(COGSCI_CONFIG.values())[0]['num_experiments']}", "yellow"))
print(colored(f"Total number of trials per seededrun: {len(global_trial_names)} (E1 to E50)", "yellow"))

for SETUP_IDX, (setup_name,setup_params) in enumerate(COGSCI_CONFIG.items()):
    print(colored(f"Running setup {SETUP_IDX+1}: {setup_name}", "cyan"))
    try:
        ESS_threshold = setup_params['ESS_threshold']
        movement_noise = setup_params['movement_noise']
        collision_noise = setup_params['collision_noise']
        speed_noise = setup_params['speed_noise']
        movement_noise_occ = setup_params['movement_noise_occ']
        collision_noise_occ = setup_params['collision_noise_occ']
        speed_noise_occ = setup_params['speed_noise_occ']
        flip_prob = setup_params['flip_prob']
        num_particles = setup_params['num_particles']
        num_experiments = setup_params['num_experiments']
        dir_stepprop_noise = setup_params['dir_stepprop_noise']
        random_number = random.randint(0,1e9)
        
        Model_Input.update('σ_NOCOL_dir_sim', d2r(movement_noise))
        Model_Input.update('σ_COL_dir_sim', d2r(collision_noise))
        Model_Input.update('σ_speed_sim', jnp.float32(speed_noise))
        Model_Input.update('σ_NOCOL_dir_occ', d2r(movement_noise_occ))
        Model_Input.update('σ_COL_dir_occ', d2r(collision_noise_occ))
        Model_Input.update('σ_speed_occ', jnp.float32(speed_noise_occ))
        Model_Input.update('flip_prob', jnp.float32(flip_prob))
        Model_Input.update('σ_NOCOL_dir_stepprop', d2r(dir_stepprop_noise))
        Model_Input.prepare_input()

        key = jax.random.PRNGKey(random_number)

        ALL_JTAP_data = {}


        for TRIAL_IDX in tqdm(range(num_trials), desc="Trial Number"):
            key, _ = jax.random.split(key,2)
            init_proposal_args = make_init_proposal_args_rg(Model_Input, observations_np[TRIAL_IDX][0])
            init_inference_chm = make_init_chm_rg(Model_Input, observations_np[TRIAL_IDX][0])
            num_t_steps = trial_frame_counts[TRIAL_IDX]
            padded_obs = padded_observations[TRIAL_IDX]
            key, exp_key = jax.random.split(key,2)
            start_time = time.time()
            try:
                exp_keys = jax.random.split(exp_key, num_experiments)
                JTAP_data_ = sequential_monte_carlo_rg_vmap(exp_keys, Model_Input, ESS_threshold, 
                    init_proposal_args, init_inference_chm, 
                    padded_obs, max_num_frames, num_particles
                )
            except Exception as e:
                print(f"Function raised an exception: {e}")
                continue

            JTAP_data ={
                'weights' : JTAP_data_['weights'][:,:num_t_steps,:],
                'obs_is_fully_hidden' : JTAP_data_['obs_is_fully_hidden'][:,:num_t_steps],
                'prediction' : {'rg' :JTAP_data_['prediction']['rg'][:,:num_t_steps,...]},
                'resampled' : JTAP_data_['resampled'][:, :num_t_steps],
                'ESS' : JTAP_data_['ESS'][:, :num_t_steps],
            }

            ALL_JTAP_data[global_trial_names[TRIAL_IDX]] = JTAP_data

    except Exception as e:
        print(f"Setup {setup_name} raised an exception: {e}")
        continue

    print(colored(f"Computing metrics for the {setup_name} setup", 'cyan'))

    pseudo_participant_multiplier = 100
    decay_T = 20 # 2.67s delay (At 30FPS, skipping 4 frames, 20 frames is (20/30)*4 = 2.67s)

    # FIRST WE START WITH THE TARGETED ANALYSIS WHERE WE ONLY USE THE FRAMES WITH OCCLUSION
    # NOTE: THAT WE ARE FIRST USING TRIALS WITH OCCLUSION ONLY
    ALL_stacked_raw_beliefs, ALL_stacked_frozen_baseline, ALL_stacked_decayed_baseline,\
    model_beliefs, frozen_beliefs, decayed_beliefs = get_raw_beliefs(ALL_JTAP_data, trials_with_occlusion,\
        trial_frame_counts, trial_name_to_idx, occlusion_frames, decay_T = decay_T)    

    # Define parameters for each truncated normal distribution as tuples
    theta_press_params = (0.6, 0.5, 0.35, 0.85)  # mean, std, lower, upper
    tau_press_params = (2.0, 0.01, jnp.arange(2, 3))
    theta_release_params = (0.25, 0.10, 0.15, 0.35)
    tau_hold_params = (3.0, 1.0, jnp.arange(2, 6))
    tau_delay_params = (2.5, 0.5, jnp.arange(2, 4))
    init_tau_delay_params = (7.0, 4.0, jnp.arange(5, 10))

    keypress_data, keydist_data, stacked_score_data, score_data, switch_data = beliefs_to_keypress_and_scores(
        None, None, None, None, None, None,
        theta_press_params, tau_press_params, theta_release_params, tau_hold_params, tau_delay_params, init_tau_delay_params,
        ALL_stacked_raw_beliefs, ALL_stacked_frozen_baseline, ALL_stacked_decayed_baseline, trial_df, pseudo_participant_multiplier,
        disable_score=False, sample_normal=True, sample_from_beliefs = False, no_press_thresh = False, use_net_evidence = False, equal_thresh = True
    )

    ALL_stacked_key_presses, ALL_stacked_key_presses_BASELINE_frozen, ALL_stacked_key_presses_BASELINE_decayed = keypress_data
    ALL_stacked_key_dist, ALL_stacked_key_dist_BASELINE_frozen, ALL_stacked_key_dist_BASELINE_decayed = keydist_data
    ALL_stacked_scores, ALL_stacked_scores_BASELINE_frozen, ALL_stacked_scores_BASELINE_decayed = stacked_score_data
    scores_model, scores_BASELINE_frozen, scores_BASELINE_decayed = score_data
    model_stacked_key_SWITCHES, frozen_stacked_key_SWITCHES, decayed_stacked_key_SWITCHES = switch_data

    keypress_dist_over_time_model, keypress_dist_over_time_frozen, keypress_dist_over_time_decayed, keypress_dist_over_time_HUMAN, \
    DECISION_DIST_PRESSED_BUTTON_model, DECISION_DIST_PRESSED_BUTTON_frozen, DECISION_DIST_PRESSED_BUTTON_decayed, DECISION_DIST_PRESSED_BUTTON_HUMAN, \
    _, _, _, _, _, _, \
    correl_weights_model, correl_weights_frozen, correl_weights_decayed, \
    valid_model_conditional_green, valid_frozen_conditional_green, valid_decayed_conditional_green, \
    valid_human_conditional_green_v_model, valid_human_conditional_green_v_frozen, valid_human_conditional_green_v_decayed = get_decision_choice_values(
        ALL_stacked_key_dist, ALL_stacked_key_dist_BASELINE_frozen, ALL_stacked_key_dist_BASELINE_decayed, HUMAN_stacked_key_dist, HUMAN_stacked_key_presses, occlusion_frames = occlusion_frames)
    
    # SEPARATE THE NAMES FOR THE TARGETED ANALYSIS
    if setup_name == 'JTAP':
        keypress_dist_over_time_model_model_setup = keypress_dist_over_time_model
        DECISION_DIST_PRESSED_BUTTON_model_model_setup = DECISION_DIST_PRESSED_BUTTON_model
        valid_model_conditional_green_model_setup = valid_model_conditional_green
        valid_human_conditional_green_v_model_model_setup = valid_human_conditional_green_v_model
        correl_weights_model_model_setup = correl_weights_model
        scores_model_model_setup = scores_model
        ALL_stacked_key_dist_model_model_setup = ALL_stacked_key_dist
        ALL_stacked_key_presses_model_model_setup = ALL_stacked_key_presses
        ALL_stacked_raw_beliefs_model_model_setup = ALL_stacked_raw_beliefs
        model_stacked_key_SWITCHES_model_setup = model_stacked_key_SWITCHES
        model_beliefs_model_setup = model_beliefs

    if setup_name == 'Frozen_Ablation':
        keypress_dist_over_time_frozen_frozen_setup = keypress_dist_over_time_frozen
        DECISION_DIST_PRESSED_BUTTON_frozen_frozen_setup = DECISION_DIST_PRESSED_BUTTON_frozen
        valid_frozen_conditional_green_frozen_setup = valid_frozen_conditional_green
        valid_human_conditional_green_v_frozen_frozen_setup = valid_human_conditional_green_v_frozen
        correl_weights_frozen_frozen_setup = correl_weights_frozen
        scores_BASELINE_frozen_frozen_setup = scores_BASELINE_frozen
        ALL_stacked_key_dist_BASELINE_frozen_frozen_setup = ALL_stacked_key_dist_BASELINE_frozen
        ALL_stacked_key_presses_BASELINE_frozen_frozen_setup = ALL_stacked_key_presses_BASELINE_frozen
        ALL_stacked_raw_beliefs_frozen_frozen_setup = ALL_stacked_frozen_baseline
        frozen_stacked_key_SWITCHES_frozen_setup = frozen_stacked_key_SWITCHES
        frozen_beliefs_frozen_setup = frozen_beliefs

    if setup_name == 'Decaying_Ablation':
        keypress_dist_over_time_decayed_decayed_setup = keypress_dist_over_time_decayed
        DECISION_DIST_PRESSED_BUTTON_decayed_decayed_setup = DECISION_DIST_PRESSED_BUTTON_decayed
        valid_decayed_conditional_green_decayed_setup = valid_decayed_conditional_green
        valid_human_conditional_green_v_decayed_decayed_setup = valid_human_conditional_green_v_decayed
        correl_weights_decayed_decayed_setup = correl_weights_decayed
        scores_BASELINE_decayed_decayed_setup = scores_BASELINE_decayed
        ALL_stacked_key_dist_BASELINE_decayed_decayed_setup = ALL_stacked_key_dist_BASELINE_decayed
        ALL_stacked_key_presses_BASELINE_decayed_decayed_setup = ALL_stacked_key_presses_BASELINE_decayed
        ALL_stacked_raw_beliefs_decayed_decayed_setup = ALL_stacked_decayed_baseline
        decayed_stacked_key_SWITCHES_decayed_setup = decayed_stacked_key_SWITCHES
        decayed_beliefs_decayed_setup = decayed_beliefs

    # FINALLY We dod that analysis for ALL TRIALS for just JTAP
    if setup_name == 'JTAP':
        # NOTE: NOW WE ARE USING ALL TRIALS
        ALL_stacked_raw_beliefs_all_trials, ALL_stacked_frozen_baseline_all_trials, ALL_stacked_decayed_baseline_all_trials,\
        model_beliefs_all_trials, frozen_beliefs_all_trials, decayed_beliefs_all_trials = get_raw_beliefs(ALL_JTAP_data, global_trial_names, trial_frame_counts, trial_name_to_idx, occlusion_frames, decay_T = decay_T)    
        keypress_data_all_trials, keydist_data_all_trials, stacked_score_data_all_trials, score_data_all_trials, switch_data_all_trials = beliefs_to_keypress_and_scores(
            None, None, None, None, None, None,
            theta_press_params, tau_press_params, theta_release_params, tau_hold_params, tau_delay_params, init_tau_delay_params,
            ALL_stacked_raw_beliefs_all_trials, ALL_stacked_frozen_baseline_all_trials, ALL_stacked_decayed_baseline_all_trials, trial_df, pseudo_participant_multiplier,
            disable_score=False, sample_normal=True, sample_from_beliefs = False, no_press_thresh = False, use_net_evidence = False, equal_thresh = True
        )

        ALL_stacked_key_presses_all_trials, ALL_stacked_key_presses_BASELINE_frozen_all_trials, ALL_stacked_key_presses_BASELINE_decayed_all_trials = keypress_data_all_trials
        ALL_stacked_key_dist_all_trials, ALL_stacked_key_dist_BASELINE_frozen_all_trials, ALL_stacked_key_dist_BASELINE_decayed_all_trials = keydist_data_all_trials
        ALL_stacked_scores_all_trials, ALL_stacked_scores_BASELINE_frozen_all_trials, ALL_stacked_scores_BASELINE_decayed_all_trials = stacked_score_data_all_trials
        scores_model_all_trials, scores_BASELINE_frozen_all_trials, scores_BASELINE_decayed_all_trials = score_data_all_trials
        model_stacked_key_SWITCHES_all_trials, frozen_stacked_key_SWITCHES_all_trials, decayed_stacked_key_SWITCHES_all_trials = switch_data_all_trials

        keypress_dist_over_time_model_all_trials, keypress_dist_over_time_frozen_all_trials, keypress_dist_over_time_decayed_all_trials, keypress_dist_over_time_HUMAN_all_trials, \
        DECISION_DIST_PRESSED_BUTTON_model_all_trials, DECISION_DIST_PRESSED_BUTTON_frozen_all_trials, DECISION_DIST_PRESSED_BUTTON_decayed_all_trials, DECISION_DIST_PRESSED_BUTTON_HUMAN_all_trials, \
        _, _, _, _, _, _, \
        correl_weights_model_all_trials, correl_weights_frozen_all_trials, correl_weights_decayed_all_trials, \
        valid_model_conditional_green_all_trials, valid_frozen_conditional_green_all_trials, valid_decayed_conditional_green_all_trials, \
        valid_human_conditional_green_v_model_all_trials, valid_human_conditional_green_v_frozen_all_trials, valid_human_conditional_green_v_decayed_all_trials = get_decision_choice_values(
            ALL_stacked_key_dist_all_trials, ALL_stacked_key_dist_BASELINE_frozen_all_trials, ALL_stacked_key_dist_BASELINE_decayed_all_trials, HUMAN_stacked_key_dist, HUMAN_stacked_key_presses, occlusion_frames = None)
        
print(colored("Generating Log Frequency plot for JTAP (all trials and frames)", "cyan"))
modelfreq_fig = create_log_frequency_heatmaps_model_only(
    DECISION_DIST_PRESSED_BUTTON_HUMAN_all_trials, DECISION_DIST_PRESSED_BUTTON_model_all_trials,
    valid_human_conditional_green_v_model_all_trials,
    valid_model_conditional_green_all_trials, 
    correl_weights_model=correl_weights_model_all_trials,
    bins=25, weighted=False, cmap='Oranges', cmap_reverse=False, model_name='JTAP'
)
modelfreq_fig.savefig("jtap_all_trials.pdf", format="pdf", bbox_inches="tight")

print(colored("Generating Log Frequency plot for targeted analysis, comparing JTAP with Frozen and Decaying", "cyan"))

logfreq_fig = create_log_frequency_heatmaps(
    DECISION_DIST_PRESSED_BUTTON_HUMAN, DECISION_DIST_PRESSED_BUTTON_model_model_setup, DECISION_DIST_PRESSED_BUTTON_frozen_frozen_setup, 
    DECISION_DIST_PRESSED_BUTTON_decayed_decayed_setup, valid_human_conditional_green_v_model_model_setup, valid_human_conditional_green_v_frozen_frozen_setup,
    valid_human_conditional_green_v_decayed_decayed_setup, valid_model_conditional_green_model_setup, valid_frozen_conditional_green_frozen_setup,
    valid_decayed_conditional_green_decayed_setup, correl_weights_model_model_setup, correl_weights_frozen_frozen_setup, correl_weights_decayed_decayed_setup, bins = 25,
    weighted = False, cmap = 'Oranges', cmap_reverse = False, model_name = 'JTAP', weight_scaler = 1.0
)
logfreq_fig.savefig("jtap_vs_ablations_targeted_analysis.pdf", format="pdf", bbox_inches="tight")


def make_illust_figure_new(HUMAN_stacked_key_dist, model_beliefs, frozen_beliefs, decayed_beliefs, 
                        ALL_rgb_videos_full_frames, position_data, trial_names, occlusion_frames, FPS, skip_t = 4, model_name = 'JTAP'):
    fig = plt.figure(figsize=(44, 20), constrained_layout = True)  # Increased size for better layout
    grid = GridSpec(5, 4, figure=fig, width_ratios=[1, 1, 1, 1], height_ratios=[2, 1, 1, 1, 1])
    grid.update(hspace=0., wspace=0.05)

    example_names = ['A', 'B', 'C', 'D']

    for i, trial_name in enumerate(trial_names):
        # Determine grid positions
        row, col = divmod(i, 2)

        # Left: Image with trajectory (spanning 4 rows)
        # image_ax = fig.add_subplot(grid[4 * row:4 * row + 4, 2 * col])
        image_ax = fig.add_subplot(grid[0, i])
        image_to_edit = ALL_rgb_videos_full_frames[trial_name][0]
        image = draw_trajectory_avoiding_dark_pixels(image_to_edit, position_data[trial_name], FPS)
        image_ax.imshow(image)
        image_ax.axis('off')
        image_ax.set_title(f"Trial {example_names[i]}", fontsize=44, fontweight = 'bold',#, bbox= dict(facecolor='snow', edgecolor='black', boxstyle='round,pad=0.3'),
            pad = 15)
        
        # Add a black rectangle around the image
        rect = Rectangle((0, 0), 1, 1, linewidth=3, edgecolor="black", facecolor="none", transform=image_ax.transAxes)
        image_ax.add_patch(rect)

        # Right: 4 line plots stacked vertically
        belief_labels = ['Participants', model_name, 'Frozen', 'Decaying']
        belief_data = [
            HUMAN_stacked_key_dist[trial_name],
            model_beliefs[trial_name],
            frozen_beliefs[trial_name],
            decayed_beliefs[trial_name]
        ]
        colors = ['green', 'red', 'blue']

        for j, (label, data) in enumerate(zip(belief_labels, belief_data)):
            ax = fig.add_subplot(grid[1+j, i])  # Plot in the right 2/3 space
            time = skip_t*(np.arange(data.shape[0]))/FPS
            for k, color in enumerate(colors):
                ax.plot(time, data[:, k], label=color.capitalize(), color=color, linewidth=5)
            ax.set_xlim([0, time[-1]])
            ax.set_ylim([-0.01, 1.01])
            # ax.set_ylabel("Proportion", fontsize=27, fontweight = 'bold')
            if j == 3:
                ax.set_xlabel("Time (s)", fontsize=34, fontweight = 'bold')
            # ax.legend(fontsize=8)
                ax.tick_params(axis='x', which='major', labelsize=30)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis='y', which='major', labelsize=30)


            if label == 'Human':
                fcolor = 'bisque'
            elif label == model_name:
                fcolor = 'lightblue'
            else:
                fcolor = 'aliceblue'

            text_style = {
                'fontsize': 36,
                'rotation': 90,
                'va': 'center',
                'ha': 'center',
                'fontweight': 'bold',
                'color': 'black',
                # 'bbox': dict(facecolor=fcolor, edgecolor='black', boxstyle='round,pad=0.15', linewidth=0.5)
                # 'bbox': dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', linewidth=0.5)
            }
            
            # Add title on the right side of the plot
            ax.text(-0.115, 0.5, label, transform=ax.transAxes, **text_style)
            ax.grid(visible=True, which='major', color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
            ax.grid(visible=True, which='minor', color='lightgray', linestyle=':', linewidth=0.5, alpha=0.5)

    # for divider in [0.25, 0.75]:
    # fig.add_artist(Line2D([0.1, 0.91], [0.4875, 0.4875], transform=fig.transFigure, color="black", linewidth=2, linestyle="--"))
    fig.add_artist(Line2D([0.5, 0.5], [0.05, 0.95], transform=fig.transFigure, color="black", linewidth=2, linestyle="--"))
    fig.add_artist(Line2D([0.25, 0.25], [0.05, 0.8], transform=fig.transFigure, color="black", linewidth=2, linestyle="--"))
    fig.add_artist(Line2D([0.75, 0.75], [0.05, 0.95], transform=fig.transFigure, color="black", linewidth=2, linestyle="--"))

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color='red', lw=5, label='Red'),
        Line2D([0], [1], color='green', lw=5, label='Green'),
        Line2D([0], [2], color='blue', lw=5, label='Uncertain')
    ]

    font_properties = font_manager.FontProperties(weight='bold', size = 34)

    fig.legend(
        handles=legend_elements, loc='upper left', bbox_to_anchor=(0.21125, 0.95), prop=font_properties, 
        title=None,  frameon=True, shadow=True, edgecolor='black', ncol=1, fancybox=True
    )

    # plt.tight_layout()
    return fig

illust_fig = make_illust_figure_new(HUMAN_stacked_key_dist, ALL_stacked_key_dist_model_model_setup, ALL_stacked_key_dist_BASELINE_frozen_frozen_setup, ALL_stacked_key_dist_BASELINE_decayed_decayed_setup, 
                                ALL_rgb_videos_full_frames, position_data, ['E3', 'E19', 'E33', 'E7'], occlusion_frames, FPS = 30)
illust_fig.show()

print(colored("Saving illustrative example figure...", "cyan"))
illust_fig.savefig("illustrative_examples.pdf", format="pdf", bbox_inches="tight")

# Rater reliability

human_scores_ = np.array([x for x in HUMAN_stacked_scores.values()])
num_participants = human_scores_.shape[1]
num_trials_ = human_scores_.shape[0]
df = pd.DataFrame(human_scores_, columns=[f'P{i+1}' for i in range(num_participants)], index=[f'Trial{i+1}' for i in range(num_trials_)])
long_df = pd.melt(df.reset_index(), id_vars=['index'], value_vars=df.columns,
                  var_name='participant', value_name='score')
long_df.rename(columns={'index': 'trial'}, inplace=True)
icc = pg.intraclass_corr(data=long_df, targets='trial', raters='participant', ratings='score')

print(colored("Analyzing human inter-rater reliability...", "cyan"))
print(colored(f"Intraclass Correlation Coefficient (ICC1k): {icc.loc[icc['Type'] == 'ICC1k', 'ICC'].iloc[0]:.4f}", "green"))