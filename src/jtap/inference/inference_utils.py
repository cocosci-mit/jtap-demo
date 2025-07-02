import copy
import cv2
import os
import pickle
import warnings
import pandas as pd

import jax
import jax.numpy as jnp
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax.scipy.special import logsumexp
from matplotlib import rcParams, font_manager
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import (
    t, ttest_ind, ttest_1samp, pearsonr, truncnorm,
    NearConstantInputWarning, ConstantInputWarning
)
from tqdm import tqdm

from jtap.utils import get_fonts_dir

# Font setup
font_path = os.path.join(get_fonts_dir(), "DMSans-Regular.ttf")
dm_sans_prop = font_manager.FontProperties(fname=font_path)
font_manager.fontManager.addfont(font_path)
font_manager.fontManager.addfont(os.path.join(get_fonts_dir(), "DMSans-Bold.ttf"))
rcParams['font.family'] = dm_sans_prop.get_name()

# Warning filters
warnings.filterwarnings("ignore", category=NearConstantInputWarning)
warnings.filterwarnings("ignore", category=ConstantInputWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

check_invalid = lambda x: jnp.logical_or(jnp.isnan(x), jnp.isinf(x))

def extract_disjoint_rectangles(mask, scale = 0.1):
    rectangles = []
    visited = np.zeros_like(mask, dtype=bool)
    labeled_mask = np.zeros_like(mask, dtype=int)  # Array to hold labeled rectangles
    n, m = mask.shape
    label = 1  # Start labeling rectangles from 1

    def find_rectangle(x, y):
        """Find the width and height of the rectangle starting at (x, y)."""
        # Determine width of the rectangle by expanding horizontally
        width = 0
        while y + width < m and mask[x, y + width] and not visited[x, y + width]:
            width += 1

        # Determine height by expanding vertically
        height = 0
        while x + height < n and all(mask[x + height, y:y + width]) and not any(visited[x + height, y:y + width]):
            height += 1

        # Mark the found rectangle as visited and label it
        for i in range(x, x + height):
            for j in range(y, y + width):
                visited[i, j] = True
                labeled_mask[i, j] = label

        return width, height

    # Iterate over each cell in the mask
    for i in range(n):
        for j in range(m):
            if mask[i, j] and not visited[i, j]:
                width, height = find_rectangle(i, j)
                rectangles.append((j * scale, scale * (n - i - height), scale*width, scale*height))
                label += 1  # Increment label for the next rectangle

    return rectangles


# stats utils
normalize_log_weights = lambda log_weights : log_weights - logsumexp(log_weights)
effective_sample_size = lambda log_weights : jnp.exp(-logsumexp(2. * normalize_log_weights(log_weights)))

def determine_sensor_output_from_mo(mo):
    
    x = mo.x
    y = mo.y
    size = mo.size

    red_x, red_y, red_size_x, red_size_y = mo.red_sensor
    green_x, green_y, green_size_x, green_size_y = mo.green_sensor
   
    in_red = jnp.logical_not(
            jnp.any(
                jnp.array([
                    jnp.less_equal(x+size, red_x),
                    jnp.greater_equal(y, red_y + red_size_y),
                    jnp.greater_equal(x, red_x + red_size_x),
                    jnp.less_equal(y+size, red_y),
                ])
            )
        )
    
    in_green = jnp.logical_not(
            jnp.any(
                jnp.array([
                    jnp.less_equal(x+size, green_x),
                    jnp.greater_equal(y, green_y + green_size_y),
                    jnp.greater_equal(x, green_x + green_size_x),
                    jnp.less_equal(y+size, green_y),
                ])
            )
        )

    return jnp.where(jnp.logical_or(in_green, in_red), jnp.where(in_red, jnp.int8(1), jnp.int8(2)), jnp.int8(0))

determine_sensor_output_from_mo_vmap = jax.vmap(determine_sensor_output_from_mo)

def get_rg_expectation_per_t(weights_t, JTAP_prediction_data_per_t, prediction_t_offset):
    normalized_probs = jnp.exp(weights_t - logsumexp(weights_t))
    coded_rg_hits = JTAP_prediction_data_per_t['rg'][prediction_t_offset - 1, :]
    uncertain_prob = jnp.sum((coded_rg_hits == 0) * normalized_probs) / jnp.sum(normalized_probs)
    red_prob = jnp.sum((coded_rg_hits == 1) * normalized_probs) / jnp.sum(normalized_probs)
    green_prob = jnp.sum((coded_rg_hits == 2) * normalized_probs) / jnp.sum(normalized_probs)
    return green_prob, red_prob, uncertain_prob

get_rg_expectation_per_t_vmap = jax.jit(jax.vmap(get_rg_expectation_per_t, in_axes = (0,0,None)))

def get_rg_expectation(JTAP_data, prediction_t_offset):

    green_probs, red_probs, uncertain_probs = get_rg_expectation_per_t_vmap(
        JTAP_data['weights'],
        JTAP_data['prediction'],
        prediction_t_offset
    )
    return jnp.stack([green_probs, red_probs, uncertain_probs], axis=1)

get_rg_expectation_over_multiple_runs = jax.jit(jax.vmap(get_rg_expectation, in_axes = (0,None)))

# to keep observations a fixed size for the sake of JITTed inference
def pad_obs_with_last_frame(array, M):
    last_frame = array[-1]
    repeated_frames = jnp.repeat(last_frame[jnp.newaxis, ...], M, axis=0)
    extended_array = jnp.concatenate([array, repeated_frames], axis=0)
    return extended_array



def beliefs_to_keypress_and_scores(
    theta_press_range, tau_press_range, theta_release_range, tau_hold_range, tau_delay_range, init_tau_delay_range,
    theta_press_params, tau_press_params, theta_release_params, tau_hold_params, tau_delay_params, init_tau_delay_params,
    ALL_stacked_raw_beliefs, ALL_stacked_frozen_baseline, ALL_stacked_decayed_baseline, trial_df, pseudo_participant_multiplier, 
    disable_score=False, sample_normal=False, sample_from_beliefs = False, no_press_thresh = False, use_net_evidence = False, equal_thresh = False
):
    """
    Function to sample and calculate keypress and scores based on beliefs.
    """

    num_trials = len(ALL_stacked_raw_beliefs)  # number of trials
    num_pseudo_participants = list(ALL_stacked_raw_beliefs.values())[0].shape[0]

    # Uniform or normal sampling based on `sample_normal`
    if sample_normal:
        # Sample from truncated normal distribution using (mean, std, lower, upper) parameters
        random_theta_press_draws = truncated_normal_sample(*theta_press_params, num_trials*num_pseudo_participants*pseudo_participant_multiplier).reshape(num_trials, num_pseudo_participants*pseudo_participant_multiplier)
        random_tau_press_draws = discrete_normal_sample(*tau_press_params, num_trials*num_pseudo_participants*pseudo_participant_multiplier).reshape(num_trials, num_pseudo_participants*pseudo_participant_multiplier)
        random_theta_release_draws = truncated_normal_sample(*theta_release_params, num_trials*num_pseudo_participants*pseudo_participant_multiplier).reshape(num_trials, num_pseudo_participants*pseudo_participant_multiplier)
        random_tau_hold_draws = discrete_normal_sample(*tau_hold_params, num_trials*num_pseudo_participants*pseudo_participant_multiplier).reshape(num_trials, num_pseudo_participants*pseudo_participant_multiplier)
        random_tau_delay_draws = discrete_normal_sample(*tau_delay_params, num_trials*num_pseudo_participants*pseudo_participant_multiplier).reshape(num_trials, num_pseudo_participants*pseudo_participant_multiplier)
        random_init_tau_delay_draws = discrete_normal_sample(*init_tau_delay_params, num_trials*num_pseudo_participants*pseudo_participant_multiplier).reshape(num_trials, num_pseudo_participants*pseudo_participant_multiplier)
    else:
        # Uniform sampling as before
        random_theta_press_draws = np.random.choice(theta_press_range, num_trials*num_pseudo_participants*pseudo_participant_multiplier, replace=True).reshape(num_trials, num_pseudo_participants*pseudo_participant_multiplier)
        random_tau_press_draws = np.random.choice(tau_press_range, num_trials*num_pseudo_participants*pseudo_participant_multiplier, replace=True).reshape(num_trials, num_pseudo_participants*pseudo_participant_multiplier)
        random_theta_release_draws = np.random.choice(theta_release_range, num_trials*num_pseudo_participants*pseudo_participant_multiplier, replace=True).reshape(num_trials, num_pseudo_participants*pseudo_participant_multiplier)
        random_tau_hold_draws = np.random.choice(tau_hold_range, num_trials*num_pseudo_participants*pseudo_participant_multiplier, replace=True).reshape(num_trials, num_pseudo_participants*pseudo_participant_multiplier)
        random_tau_delay_draws = np.random.choice(tau_delay_range, num_trials*num_pseudo_participants*pseudo_participant_multiplier, replace=True).reshape(num_trials, num_pseudo_participants*pseudo_participant_multiplier)
        random_init_tau_delay_draws = np.random.choice(init_tau_delay_range, num_trials*num_pseudo_participants*pseudo_participant_multiplier, replace=True).reshape(num_trials, num_pseudo_participants*pseudo_participant_multiplier)


    if equal_thresh:
        random_theta_release_draws = copy.deepcopy(random_theta_press_draws)

    key = jax.random.PRNGKey(np.random.randint(1e9))

    ALL_stacked_key_presses = raw_beliefs_to_keypresses(
        ALL_stacked_raw_beliefs, pseudo_participant_multiplier,
        random_theta_press_draws, random_tau_press_draws, random_theta_release_draws,
        random_tau_hold_draws, random_tau_delay_draws, random_init_tau_delay_draws,
        sample_from_beliefs, key, no_press_thresh, use_net_evidence
    )

    ALL_stacked_key_dist = {k:get_rg_distribution(v) for k,v in ALL_stacked_key_presses.items()}

    ALL_stacked_key_presses_BASELINE_frozen = raw_beliefs_to_keypresses(
        ALL_stacked_frozen_baseline, pseudo_participant_multiplier, 
        random_theta_press_draws, random_tau_press_draws, random_theta_release_draws, 
        random_tau_hold_draws, random_tau_delay_draws, random_init_tau_delay_draws,
        sample_from_beliefs, key, no_press_thresh, use_net_evidence
    )
    ALL_stacked_key_dist_BASELINE_frozen = {k:get_rg_distribution(v) for k,v in ALL_stacked_key_presses_BASELINE_frozen.items()}

    ALL_stacked_key_presses_BASELINE_decayed = raw_beliefs_to_keypresses(
        ALL_stacked_decayed_baseline, pseudo_participant_multiplier, 
        random_theta_press_draws, random_tau_press_draws, random_theta_release_draws, 
        random_tau_hold_draws, random_tau_delay_draws, random_init_tau_delay_draws,
        sample_from_beliefs, key, no_press_thresh, use_net_evidence, decay = True
    )
    ALL_stacked_key_dist_BASELINE_decayed = {k:get_rg_distribution(v) for k,v in ALL_stacked_key_presses_BASELINE_decayed.items()}

    if not disable_score:
        ALL_stacked_scores = {}
        for trial_name, stacked_key_presses in ALL_stacked_key_presses.items():
            rg_outcome_idx = trial_df[trial_df['global_trial_name'] == trial_name]['rg_outcome_idx'].tolist()[0]
            scores = get_model_score_vmap(stacked_key_presses, rg_outcome_idx)
            ALL_stacked_scores[trial_name] = scores
        scores_model = {trial_name: jnp.mean(scores) for trial_name, scores in ALL_stacked_scores.items()}

        ALL_stacked_scores_BASELINE_frozen = {}
        for trial_name, stacked_key_presses in ALL_stacked_key_presses_BASELINE_frozen.items():
            rg_outcome_idx = trial_df[trial_df['global_trial_name'] == trial_name]['rg_outcome_idx'].tolist()[0]
            scores = get_model_score_vmap(stacked_key_presses, rg_outcome_idx)
            ALL_stacked_scores_BASELINE_frozen[trial_name] = scores
        scores_BASELINE_frozen = {trial_name: jnp.mean(scores) for trial_name, scores in ALL_stacked_scores_BASELINE_frozen.items()}

        ALL_stacked_scores_BASELINE_decayed = {}
        for trial_name, stacked_key_presses in ALL_stacked_key_presses_BASELINE_decayed.items():
            rg_outcome_idx = trial_df[trial_df['global_trial_name'] == trial_name]['rg_outcome_idx'].tolist()[0]
            scores = get_model_score_vmap(stacked_key_presses, rg_outcome_idx)
            ALL_stacked_scores_BASELINE_decayed[trial_name] = scores
        scores_BASELINE_decayed = {trial_name: jnp.mean(scores) for trial_name, scores in ALL_stacked_scores_BASELINE_decayed.items()}

    model_stacked_key_SWITCHES = {k:jnp.sum(v[:,1:] != v[:,:-1], axis = 1) for k, v in ALL_stacked_key_presses.items()}
    frozen_stacked_key_SWITCHES = {k:jnp.sum(v[:,1:] != v[:,:-1], axis = 1) for k, v in ALL_stacked_key_presses_BASELINE_frozen.items()}
    decayed_stacked_key_SWITCHES = {k:jnp.sum(v[:,1:] != v[:,:-1], axis = 1) for k, v in ALL_stacked_key_presses_BASELINE_decayed.items()}

    if disable_score:
        return (ALL_stacked_key_presses, ALL_stacked_key_presses_BASELINE_frozen, ALL_stacked_key_presses_BASELINE_decayed),\
    (ALL_stacked_key_dist, ALL_stacked_key_dist_BASELINE_frozen, ALL_stacked_key_dist_BASELINE_decayed),\
    (model_stacked_key_SWITCHES, frozen_stacked_key_SWITCHES, decayed_stacked_key_SWITCHES)
    else:
        return (ALL_stacked_key_presses, ALL_stacked_key_presses_BASELINE_frozen, ALL_stacked_key_presses_BASELINE_decayed),\
        (ALL_stacked_key_dist, ALL_stacked_key_dist_BASELINE_frozen, ALL_stacked_key_dist_BASELINE_decayed),\
        (ALL_stacked_scores, ALL_stacked_scores_BASELINE_frozen, ALL_stacked_scores_BASELINE_decayed),\
        (scores_model, scores_BASELINE_frozen, scores_BASELINE_decayed),\
        (model_stacked_key_SWITCHES, frozen_stacked_key_SWITCHES, decayed_stacked_key_SWITCHES)


def key_press_decision_model(raw_beliefs, 
                             theta_press, 
                             tau_press, 
                             theta_release, 
                             tau_hold, 
                             tau_delay,
                             init_tau_delay,
                             sample_from_beliefs = False,
                             initkey = None,
                             no_press_thresh = False,
                             decay = False):

    T, _ = raw_beliefs.shape
    # Initialize outputs and state
    button_pressed = 2  # 2: No button, 1: Red, 2: Green
    hysteresis_counter = 0  # Counter to track how long the button has been held
    red_evidence_accum_counter, green_evidence_accum_counter = 0,0

    def step(carry, t):
        # adjust t with time delay # CRUCIAL
        t_adjusted = t - tau_delay

        key, button_pressed, hysteresis_counter, red_evidence_accum_counter, green_evidence_accum_counter = carry
        key, next_key = jax.random.split(key, 2)

        def sample_from_belief(key, red_bel, green_bel, uncertain_bel, red_evidence_accum_counter, green_evidence_accum_counter, epsilon_ = 1e-10):
            log_beliefs = jnp.log(jnp.clip(jnp.array([green_bel, red_bel, uncertain_bel]), epsilon_, None))
            choice = jax.random.categorical(key, log_beliefs)
            # jprint("beliefs: {}, choice: {}", [green_bel, red_bel], choice)
            return (jnp.int32(choice), 0, red_evidence_accum_counter, green_evidence_accum_counter)

        # Determine behavior based on whether a button is pressed
        def should_i_press(key, hysteresis_counter, red_evidence_accum_counter, green_evidence_accum_counter, is_pressed):

            current_red_belief = raw_beliefs[t_adjusted, 1]
            current_green_belief = raw_beliefs[t_adjusted, 0]
            current_uncertain_belief = raw_beliefs[t_adjusted, 2]

            red_crosses_threshold = current_red_belief >= theta_press
            green_crosses_threshold = current_green_belief >= theta_press

            red_evidence_accum_counter = jnp.where(red_crosses_threshold, red_evidence_accum_counter + 1, 0)
            green_evidence_accum_counter = jnp.where(green_crosses_threshold, green_evidence_accum_counter + 1, 0)

            should_press_red = jnp.greater_equal(red_evidence_accum_counter, tau_press)
            should_press_green = jnp.greater_equal(green_evidence_accum_counter, tau_press)


            # hysteresis
            should_press_red, should_press_green = jax.lax.cond(
                is_pressed,
                lambda: jax.lax.cond(
                    jnp.logical_and(jnp.logical_not(decay), jnp.less(current_uncertain_belief, 0.99)),
                    lambda : (True, True),
                    lambda: (should_press_red, should_press_green)
                ),
                lambda: (should_press_red, should_press_green)
            )

            current_uncertain_belief_ = jnp.where(no_press_thresh, current_uncertain_belief, 0.0)

            sample_a_button = jnp.logical_and(
                jnp.logical_or(jnp.logical_or(should_press_red, should_press_green),no_press_thresh),
                sample_from_beliefs
            )
            
            return jax.lax.cond(sample_a_button,
                lambda: sample_from_belief(key, current_red_belief, current_green_belief, current_uncertain_belief_, red_evidence_accum_counter, green_evidence_accum_counter),
                lambda: jax.lax.cond(
                    # if both cross threshold, press the one with higher belief
                    jnp.logical_and(should_press_red, should_press_green),
                        lambda : jax.lax.cond(current_red_belief > current_green_belief,
                            lambda : (1, 0, red_evidence_accum_counter, green_evidence_accum_counter),
                            lambda : (0, 0, red_evidence_accum_counter, green_evidence_accum_counter),
                        ),
                    lambda : jax.lax.cond(should_press_red,
                        lambda : (1, 0, red_evidence_accum_counter, green_evidence_accum_counter),
                        lambda : jax.lax.cond(should_press_green,
                            lambda : (0, 0, red_evidence_accum_counter, green_evidence_accum_counter),
                            lambda : (2, 0, red_evidence_accum_counter, green_evidence_accum_counter),
                        )
                    )
                )
            )

        def should_i_hold(key, hysteresis_counter, red_evidence_accum_counter, green_evidence_accum_counter, is_pressed):

            current_belief = raw_beliefs[t_adjusted, button_pressed]
            # update hysteresis counter
            hysteresis_counter += 1

            hold_continue = jnp.logical_or(
                jnp.greater_equal(current_belief, theta_release),
                jnp.less_equal(hysteresis_counter,tau_hold),
            )

            return jax.lax.cond(
                hold_continue,
                lambda *_: (button_pressed, hysteresis_counter, red_evidence_accum_counter, green_evidence_accum_counter),
                should_i_press, # release condition --> check if any press condition is made
                key, hysteresis_counter, red_evidence_accum_counter, green_evidence_accum_counter, is_pressed
            )

        # Use cond to switch between cases
        new_button_pressed, new_hysteresis_counter, red_evidence_accum_counter, green_evidence_accum_counter = jax.lax.cond(
            jnp.less(t, init_tau_delay),
            lambda : (2, 0, red_evidence_accum_counter, green_evidence_accum_counter), # If initial time delay has not passed, dont press yet
            lambda : jax.lax.cond(
                button_pressed == 2,
                should_i_press,
                should_i_hold,
                key, hysteresis_counter, red_evidence_accum_counter, green_evidence_accum_counter, button_pressed != 2
            )
        )

        # jprint("Timestep: {}, Button Pressed: {}, Hold Counter: {}", t, new_button_pressed, new_hysteresis_counter)

        return (next_key, new_button_pressed, new_hysteresis_counter, red_evidence_accum_counter, green_evidence_accum_counter), new_button_pressed

    # Iterate over timesteps
    _, stacked_decisions = jax.lax.scan(
        step, (initkey, button_pressed, hysteresis_counter, red_evidence_accum_counter, green_evidence_accum_counter), jnp.arange(T)
    )

    return stacked_decisions


def raw_beliefs_to_keypresses(ALL_stacked_raw_beliefs, pseudo_participant_multiplier, 
        random_theta_press_draws, random_tau_press_draws, random_theta_release_draws, 
        random_tau_hold_draws, random_tau_delay_draws, random_init_tau_delay_draws,
        sample_from_beliefs = False, key = None, no_press_thresh = False, use_net_evidence = False, decay = False):
    # Find maximum T and trial names
    trial_names = list(ALL_stacked_raw_beliefs.keys())
    max_T = max(v.shape[1] for v in ALL_stacked_raw_beliefs.values())
    
    # Pad beliefs and store original lengths
    padded_beliefs = [
        jnp.pad(v, ((0, 0), (0, max_T - v.shape[1]), (0, 0)), constant_values=0)
        for v in ALL_stacked_raw_beliefs.values()
    ]
    original_lengths = [v.shape[1] for v in ALL_stacked_raw_beliefs.values()]
    
    # Stack and process with vmap
    stacked_padded_beliefs = jnp.stack(padded_beliefs)

    # Tile the stacked padded beliefs for the multiplier
    tiled_stacked_padded_beliefs = jnp.tile(stacked_padded_beliefs, (1,pseudo_participant_multiplier,1,1))
    keys = jax.random.split(key, jnp.size(random_theta_press_draws)).reshape((*random_theta_press_draws.shape, 2))

    key_press_decision_model_double_vmap = jax.vmap(jax.vmap(key_press_decision_model, in_axes = (0,0,0,0,0,0,0,None,0,None,None)), in_axes = (0,0,0,0,0,0,0,None,0,None,None))

    stacked_key_presses = key_press_decision_model_double_vmap(tiled_stacked_padded_beliefs, random_theta_press_draws, 
        random_tau_press_draws, random_theta_release_draws, random_tau_hold_draws, random_tau_delay_draws, random_init_tau_delay_draws,
        sample_from_beliefs, keys, no_press_thresh, decay)

    # Recover original shapes
    return {
        trial_names[i]: stacked_key_presses[i, :, :original_lengths[i]]
        for i in range(len(trial_names))
    }


def decay_distribution_exponential(probs, target_index, T, t):
    """
    Apply exponential decay to a probability distribution over T steps.

    Args:
        probs: jnp.ndarray of shape (3,), the initial probability distribution (must sum to 1).
        target_index: int, the index (0, 1, or 2) of the value to increase.
        T: int, the total number of steps for the decay.
        t: int, the current timestep (0 <= t <= T).

    Returns:
        jnp.ndarray of shape (3,), the adjusted probability distribution at timestep t.
    """
    # if not jnp.isclose(jnp.sum(probs), 1.0):
    #     raise ValueError("Input probabilities must sum to 1.")
    # if target_index < 0 or target_index > 2:
    #     raise ValueError("Target index must be 0, 1, or 2.")

    # Compute decay rate
    decay_rate = jnp.log(2) / T

    # Compute decayed probabilities for non-target indices
    non_target_indices = [i for i in range(3) if i != target_index]
    decayed_probs = jnp.array([
        probs[idx] * jnp.exp(-decay_rate * t) for idx in non_target_indices
    ])

    # Redistribute the lost probability to the target index
    increased_prob = 1 - jnp.sum(decayed_probs)

    # Construct the adjusted probability array
    adjusted_probs = jnp.zeros_like(probs)
    for i, idx in enumerate(non_target_indices):
        adjusted_probs = adjusted_probs.at[idx].set(decayed_probs[i])
    adjusted_probs = adjusted_probs.at[target_index].set(increased_prob)

    return adjusted_probs


def process_along_t_axis(data, mask ,decay_T):
    def step(carry, inputs):
        prev_values, occluded_len = carry
        values, should_freeze = inputs

        # Use frozen if in frozen mode; otherwise, use original values
        next_value, returned_value, next_occlued_len = jax.lax.cond(
            should_freeze,
            lambda _: (prev_values, decay_distribution_exponential(prev_values, 2, decay_T, occluded_len), occluded_len+1),
            lambda _: (values, values, 0),
            operand=None,
        )

        return (next_value, next_occlued_len), returned_value

    # Initial carry: (current_values, frozen_values)
    initial_carry = (data[0], 0)
    inputs = (data, mask)
    _, result = jax.lax.scan(step, initial_carry, inputs)
    return result

def get_baseline_beliefs(data, mask, decay_T):
    return jax.vmap(process_along_t_axis, in_axes=(0, None, None))(data, mask, decay_T)


@jax.jit
def get_model_score(keypress_data, rg_outcome):
    num_frames = keypress_data.shape[0]
    penalty_outcome = jnp.where(rg_outcome == 1, 0,1)
    score = 20 + 100*((jnp.sum(keypress_data == rg_outcome)/num_frames) - (jnp.sum(keypress_data == penalty_outcome)/num_frames))
    return score

get_model_score_vmap = jax.vmap(get_model_score, in_axes = (0,None))


def weighted_rmse(A, F, W):
    return jnp.sqrt(jnp.sum(W * (A - F)**2) / jnp.sum(W))

def get_rg_distribution(arr):
    one_hot = jax.nn.one_hot(arr, num_classes=3)  # Shape: (N, T, 3)
    counts = jnp.sum(one_hot, axis=0)  # Shape: (T, 3)
    normalized = counts / jnp.sum(counts, axis=1, keepdims=True)  # Shape: (T, 3)
    return normalized


def create_log_frequency_heatmaps(
    human_pressed_button, model_pressed_button, frozen_pressed_button, decayed_pressed_button,
    human_choose_green_v_model, human_choose_green_v_frozen, human_choose_green_v_decayed,
    model_choose_green, frozen_choose_green, decayed_choose_green,
    correl_weights_model=None, correl_weights_frozen=None, correl_weights_decayed=None,
    bins=20, weighted=False, cmap='viridis', cmap_reverse=False, weight_scaler=1, model_name='Model'
):
    """
    Create 6 log-frequency heatmaps arranged in a 3x2 grid comparing human and model data.
    
    Args:
        human_pressed_button, model_pressed_button, frozen_pressed_button, decayed_pressed_button:
            Arrays for P(Red or Green).
        human_choose_green_*, *_choose_green: Arrays for P(Green | Red or Green).
        bins: Number of bins for the heatmap (default: 20).
        weighted: Whether to use correlation weights (default: False).
        cmap: Matplotlib colormap name (default: 'viridis').
        cmap_reverse: Whether to reverse the colormap (default: False).
    
    Returns:
        matplotlib.figure.Figure: The complete figure with all heatmaps.
    """
    # Set style
    plt.style.use('default')
    sns.set_style("white")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(14, 22))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.15)
    
    # Define bin edges
    bin_edges = np.linspace(0, 1, bins + 1)
    
    def calculate_heatmap(human_data, model_data, correl_weights):
        # Digitize both arrays into bins

        human_bin_idx = np.digitize(human_data, bin_edges) - 1
        human_bin_idx[human_bin_idx == bins] = bins - 1
        model_bin_idx = np.digitize(model_data, bin_edges) - 1
        model_bin_idx[model_bin_idx == bins] = bins - 1
        
        # Initialize the 2D histogram
        heatmap = np.zeros((bins, bins))
        heatmap_weight = np.zeros((bins, bins))
        
        # Accumulate counts
        for h, m, weight in zip(human_bin_idx, model_bin_idx, correl_weights):
            heatmap[h, m] += 1
            heatmap_weight[h, m] += weight/59 #(from 1 to 59)
            
        # # Normalize weights
        # if np.sum(heatmap_weight) > 0:
        #     heatmap_weight /= np.sum(heatmap_weight)
            
        # Compute log frequencies
        # log_heatmap = np.log1p(heatmap) + np.log1p(heatmap_weight)
        log_heatmap = np.log1p(heatmap)
        # log_heatmap = np.log1p(heatmap_weight)
        return log_heatmap
    
    # Generate all heatmaps
    uniform_weights = np.zeros_like(human_pressed_button)
    heatmaps_pressed = [
        calculate_heatmap(human_pressed_button, model_pressed_button, uniform_weights),
        calculate_heatmap(human_pressed_button, frozen_pressed_button, uniform_weights),
        calculate_heatmap(human_pressed_button, decayed_pressed_button, uniform_weights)
    ]

    pressed_correl = [
        pearsonr(human_pressed_button, model_pressed_button)[0],
        pearsonr(human_pressed_button, frozen_pressed_button)[0],
        pearsonr(human_pressed_button, decayed_pressed_button)[0]
    ]
    
    correl_weights = [
        np.zeros_like(human_choose_green_v_model) if (correl_weights_model is None or not weighted) else correl_weights_model,
        np.zeros_like(human_choose_green_v_frozen) if (correl_weights_frozen is None or not weighted) else correl_weights_frozen,
        np.zeros_like(human_choose_green_v_decayed) if (correl_weights_decayed is None or not weighted) else correl_weights_decayed
    ]
    
    heatmaps_green = [
        calculate_heatmap(human_choose_green_v_model, model_choose_green, correl_weights[0]),
        calculate_heatmap(human_choose_green_v_frozen, frozen_choose_green, correl_weights[1]),
        calculate_heatmap(human_choose_green_v_decayed, decayed_choose_green, correl_weights[2])
    ]

    green_correl = [
        weighted_corr(human_choose_green_v_model, model_choose_green,correl_weights_model),
        weighted_corr(human_choose_green_v_frozen, frozen_choose_green, correl_weights_frozen),
        weighted_corr(human_choose_green_v_decayed, decayed_choose_green, correl_weights_decayed)
    ]
    
    # Set up colormaps
    if cmap_reverse:
        cmap = plt.get_cmap(cmap).reversed()
    else:
        cmap = plt.get_cmap(cmap)
    
    # Determine global color scale
    vmin = min(np.min(hp) for hp in heatmaps_pressed + heatmaps_green)
    vmax = max(np.max(hp) for hp in heatmaps_pressed + heatmaps_green)
    
    # Titles and labels
    titles = ['Model', 'Frozen', 'Decaying']
    column_headers = ['P(Red or Green)', 'P(Green | Red or Green)']
    column_headers = ['$\\mathbf{P(Decision)}$', '$\\mathbf{P(Green \\; | \\; Decision)}$']

    # Plot heatmaps
    for row, (title, hp, hg) in enumerate(zip(titles, heatmaps_pressed, heatmaps_green)):
        # Left heatmap (P(Red or Green))
        ax1 = fig.add_subplot(gs[row, 0])
        im1 = ax1.imshow(hp, cmap=cmap, origin='lower', extent=[0, 1, 0, 1], 
                        aspect='equal', vmin=vmin, vmax=vmax)
        # ax1.set_title(title, pad=15, fontsize=12, fontweight='bold')
        
        # Right heatmap (P(Green | Red or Green))
        ax2 = fig.add_subplot(gs[row, 1])
        im2 = ax2.imshow(hg, cmap=cmap, origin='lower', extent=[0, 1, 0, 1], 
                        aspect='equal', vmin=vmin, vmax=vmax)
        
        # Labels
        if row == 0:
            fig.text(0.5, 0.64, model_name, ha='center', va='center', 
             fontsize=26, fontweight='bold')
            # ax1.set_xlabel(model_name, fontsize=20, fontweight='bold')
            # ax2.set_xlabel(model_name, fontsize=20, fontweight='bold')
        elif row == 1:
            fig.text(0.5, 0.355, 'Frozen', ha='center', va='center', 
             fontsize=26, fontweight='bold')
            # ax1.set_xlabel('Frozen', fontsize=20, fontweight='bold')
            # ax2.set_xlabel('Frozen', fontsize=20, fontweight='bold')
        elif row == 2:
            fig.text(0.5, 0.085, 'Decaying', ha='center', va='center', 
             fontsize=26, fontweight='bold')
            # ax1.set_xlabel('Decaying', fontsize=20, fontweight='bold')
            # ax2.set_xlabel('Decaying', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Participants', fontsize=22, fontweight='bold')
        ax2.set_ylabel('Participants', fontsize=22, fontweight='bold')

        ax1.set_title(f"$r = {pressed_correl[row]:.2f}$", fontsize=24, fontweight='bold', pad = 10)
        # ax2.set_title(f"$Weighted \\; r = {green_correl[row]:.2f}$", fontsize=24, fontweight='bold')
        ax2.set_title(f"$r_{{\\text{{wtd}}}} = {green_correl[row]:.2f}$", fontsize=24, fontweight='bold', pad = 10)

        # Grid settings
        for ax in [ax1, ax2]:
            ax.grid(False)
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.set_yticks(np.linspace(0, 1, 5))
            # Add spines styling
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('#333333')
            
            # Improve tick appearance
            ax.tick_params(width=1.5, length=6, color='#333333', labelsize = 14)

        # fig.text(0.25 + row * 0.5, 1 - (row * 0.3), title, ha='center', va='center', 
        #         fontsize=14, fontweight='bold')
        
        # After plotting all heatmaps, add a single colorbar at the bottom
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])  # Adjust position as needed
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='Log Frequency', shrink=0.5)
    cbar.set_label('Log Frequency', fontsize=20)
    cbar.ax.tick_params(labelsize=14)

    # Add column headers
    fig.text(0.3, 0.91, column_headers[0], ha='center', va='center', 
             fontsize=24, fontweight='bold')
    fig.text(0.7, 0.91, column_headers[1], ha='center', va='center', 
             fontsize=24, fontweight='bold')
    
    h_line = Line2D([0.1, 0.9], [0.345, 0.345], color='black', linestyle='-', linewidth=1)
    fig.add_artist(h_line)
    h_line = Line2D([0.1, 0.9], [0.625, 0.625], color='black', linestyle='-', linewidth=1)
    fig.add_artist(h_line)
    # Add vertical line (for x = 0.5, across the figure)
    # v_line = Line2D([0.49, 0.49], [0.1, 0.9], color='black', linestyle='--', linewidth=1)
    # fig.add_artist(v_line)
    
    return fig



def draw_trajectory_avoiding_dark_pixels(image, pdata, FPS, line_color=(0, 0, 0), 
                                         line_thickness=0.5, text_color=(0, 0, 0), 
                                         font_scale=1.0, font_thickness=0.5):
    upscale_factor = 4
    high_res_image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_LINEAR)

    world_to_image_scale = 40  # Adjust scale for the upscaled image
    sorted_frames = sorted(pdata.keys())

    for i in range(1, len(sorted_frames)):
        prev_frame = sorted_frames[i - 1]
        current_frame = sorted_frames[i]

        # Convert world coordinates to image coordinates for upscaled image
        x1, y1 = (
            (pdata[prev_frame]['x'] + 0.5) * world_to_image_scale, 
            800 - (pdata[prev_frame]['y'] + 0.5) * world_to_image_scale
        )
        x2, y2 = (
            (pdata[current_frame]['x'] + 0.5) * world_to_image_scale, 
            800 - (pdata[current_frame]['y'] + 0.5) * world_to_image_scale
        )

        # Draw line segment for the trajectory
        cv2.line(
            high_res_image, 
            (int(x1), int(y1)), 
            (int(x2), int(y2)), 
            line_color, 
            int(line_thickness * upscale_factor),  # Adjust thickness for high resolution
            lineType=cv2.LINE_AA
        )

    def find_clear_position(img, x, y, text, search_radius=20):
        """Find a nearby position that avoids dark pixels, considering text size."""
        h, w, _ = img.shape
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, int(font_thickness * upscale_factor))[0]

        for r in range(1, search_radius):
            for dx, dy in [(-r, 0), (r, 0), (0, -r), (0, r)]:
                nx, ny = int(x + dx), int(y + dy)
                x_end, y_end = nx + (text_size[0] + 3), ny - (text_size[1] + 3)

                if 0 <= nx < w and 0 <= ny < h and 0 <= x_end < w and 0 <= y_end < h:
                    # Check if the entire text bounding box avoids dark pixels
                    region = img[ny - (text_size[1] + 3):ny, nx:x_end]
                    if not np.any(np.all(region < 27, axis=-1)):# and not np.any(np.all(region < 30, axis=-1)):
                        return nx, ny
        return x, y  # Default to the original position if no clear spot is found

    # Annotate seconds on the trajectory
    for frame in sorted_frames:
        if frame % FPS == 0:
            text = str(frame // FPS)
            x, y = (
                (pdata[frame]['x'] + 0.5) * world_to_image_scale, 
                800 - (pdata[frame]['y'] + 0.5) * world_to_image_scale
            )

            # Find a clear position to place the text
            x_clear, y_clear = find_clear_position(high_res_image, x, y - 15, text)

            # y_adjust = -50 if frame//FPS == 0 else 0
            y_adjust = 0

            # Annotate the frame number
            cv2.putText(
                high_res_image,
                text,  # Frame number as text
                (int(x_clear), int(y_clear - y_adjust)),  # Adjusted position
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,  # Larger font scale for high resolution
                text_color,
                thickness=int(font_thickness * upscale_factor),  # Adjust thickness for high resolution
                lineType=cv2.LINE_AA
            )

    return high_res_image


from statsmodels.stats.weightstats import DescrStatsW

def weighted_corr(x, y, weights, epsilon = 1e-8):
    # Combine x and y into 2D array


    x_std = np.std(x)
    y_std = np.std(y)    
    if x_std == 0 and y_std == 0:
        return 1.0
    
    # if x_std == 0 or y_std == 0:
    #     return 0.0
    
    
    xy = np.vstack([x, y])
    
    # Create weighted stats object
    descstats = DescrStatsW(xy.T, weights=weights)
    
    # Get correlation matrix
    corr_matrix = descstats.corrcoef
    
    # Return correlation between x and y
    return corr_matrix[0,1]




def truncated_normal_sample(mean, std, lower, upper, size):
    """
    Sample from a truncated normal distribution between lower and upper bounds.
    """
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

def discrete_normal_sample(mean, std, discrete_values, size):
    """
    Sample from a discrete approximation of a normal distribution.
    This is done by sampling from `discrete_values` with probabilities derived from the normal distribution.
    """
    # Compute probabilities for each discrete value
    probabilities = np.exp(-0.5 * ((discrete_values - mean) ** 2) / (std ** 2))
    probabilities /= probabilities.sum()  # Normalize to sum to 1
    
    return np.random.choice(discrete_values, size=size, p=probabilities)


def get_raw_beliefs(TRIAL_multiple_JTAP_data, global_trial_names, ALL_trial_frame_count, trial_name_to_idx, occlusion_frames, decay_T = 20, global_pred_len = 74):

    ALL_stacked_raw_beliefs = {}
    occ_frames = {}
    for trial_name, multiple_JTAP_data in TRIAL_multiple_JTAP_data.items():
        if trial_name in global_trial_names:
            trial_idx = trial_name_to_idx[trial_name]
            if len(occlusion_frames[trial_name]) == 0:
                occ_frames[trial_name] = jnp.zeros((ALL_trial_frame_count[trial_idx],), dtype=jnp.bool)
            else:
                occ_frames[trial_name] = jnp.zeros((ALL_trial_frame_count[trial_idx],), dtype=jnp.bool).at[jnp.array(occlusion_frames[trial_name])].set(jnp.bool_(True))
            if global_pred_len is None:
                pred_len = ALL_trial_frame_count[trial_idx]+5
            else:
                pred_len = global_pred_len
            stacked_raw_beliefs = get_rg_expectation_over_multiple_runs(multiple_JTAP_data, pred_len)[:,1:,:] # NOTE: IGNORE T = 0
            ALL_stacked_raw_beliefs[trial_name] = stacked_raw_beliefs
        
    # get frozen baseline
    ALL_stacked_frozen_baseline = {}
    for trial_name, stacked_raw_beliefs in ALL_stacked_raw_beliefs.items():
        baseline_beliefs = get_baseline_beliefs(ALL_stacked_raw_beliefs[trial_name], occ_frames[trial_name][1:],# NOTE: IGNORE T = 0
                                                decay_T=jnp.inf) # THIS IS FROZEN
        ALL_stacked_frozen_baseline[trial_name] = baseline_beliefs

    # get decayed baseline
    ALL_stacked_decayed_baseline = {}
    for trial_name, stacked_raw_beliefs in ALL_stacked_raw_beliefs.items():
        baseline_beliefs = get_baseline_beliefs(ALL_stacked_raw_beliefs[trial_name], occ_frames[trial_name][1:],# NOTE: IGNORE T = 0
                                                decay_T=decay_T) # THIS IS SET TO 10 FOR NOW
        ALL_stacked_decayed_baseline[trial_name] = baseline_beliefs

    model_beliefs = {k:jnp.mean(v,axis=0) for k,v in ALL_stacked_raw_beliefs.items()}
    frozen_beliefs = {k:jnp.mean(v,axis=0) for k,v in ALL_stacked_frozen_baseline.items()}
    decayed_beliefs = {k:jnp.mean(v,axis=0) for k,v in ALL_stacked_decayed_baseline.items()}
    return ALL_stacked_raw_beliefs, ALL_stacked_frozen_baseline, ALL_stacked_decayed_baseline, model_beliefs, frozen_beliefs, decayed_beliefs


def get_human_data(human_data_pkl_file, skip_t, global_trial_names):    
    # Load cleaned and arranged human data from the SQL database
    with open(human_data_pkl_file, "rb") as f:
        data = pickle.load(f)
        session_df = data["Session"]
        trial_df = data["Trial"]
        keystate_df = data["KeyState"]
        position_data = data["position_data"]
        occlusion_durations = data["occlusion_durations"]
        occlusion_frames_ = data["occlusion_frames"]

    occlusion_frames = {}
    for trial_name in global_trial_names:
        if trial_name in occlusion_frames_:
            occ_frames_ = occlusion_frames_[trial_name]
            occlusion_frames[trial_name] = [int(t/skip_t) for t in occ_frames_ if t % skip_t == 0]
        else:
            occlusion_frames[trial_name] = []
    reduced_keystate_df = keystate_df[keystate_df['frame'] % skip_t == 0]
    reduced_keystate_df = reduced_keystate_df[reduced_keystate_df['frame'] != 0]# NOTE: IGNORE T = 0

    reduced_position_data = {}
    for trial_name in global_trial_names:
        reduced_position_data[trial_name] = {k:v for k,v in position_data[trial_name].items() if k % skip_t == 0}

    HUMAN_stacked_key_presses = {}
    for trial_name in tqdm(global_trial_names, desc = "Calculating human key presses per trial"):
        reduced_keyframe_trial_df = reduced_keystate_df[reduced_keystate_df['global_trial_name'] == trial_name]
        num_frames = len(reduced_keyframe_trial_df['frame'].unique())
        all_keypresses = []
        for i in range(1, num_frames+1): # NOTE: IGNORE T = 0
            reduced_keyframe_trial_df_frame = reduced_keyframe_trial_df[reduced_keyframe_trial_df['frame'] == i*skip_t]
            green = jnp.array(list(reduced_keyframe_trial_df_frame['green']))
            red = jnp.array(list(reduced_keyframe_trial_df_frame['red']))
            uncertain = jnp.array(list(reduced_keyframe_trial_df_frame['uncertain']))
            keypress = 0*green + 1*red + 2*uncertain
            all_keypresses.append(keypress)
        all_keypresses = jnp.array(all_keypresses).T
        HUMAN_stacked_key_presses[trial_name] = all_keypresses

    HUMAN_stacked_key_dist = {k:get_rg_distribution(v) for k,v in HUMAN_stacked_key_presses.items()}
    HUMAN_stacked_key_SWITCHES = {k:jnp.sum(v[:,1:] != v[:,:-1], axis = 1) for k, v in HUMAN_stacked_key_presses.items()}


    HUMAN_stacked_scores = {}
    for trial_name, stacked_key_presses in HUMAN_stacked_key_presses.items():
        rg_outcome_idx = trial_df[trial_df['global_trial_name'] == trial_name]['rg_outcome_idx'].tolist()[0]
        scores = get_model_score_vmap(stacked_key_presses, rg_outcome_idx)
        HUMAN_stacked_scores[trial_name] = scores
    HUMAN_scores = {trial_name: jnp.mean(scores) for trial_name, scores in HUMAN_stacked_scores.items()}

    return session_df, trial_df, keystate_df, position_data, occlusion_durations, occlusion_frames,\
        HUMAN_stacked_key_presses, HUMAN_stacked_key_dist, HUMAN_scores, HUMAN_stacked_scores, HUMAN_stacked_key_SWITCHES


def get_decision_choice_values(ALL_stacked_key_dist, ALL_stacked_key_dist_BASELINE_frozen, 
        ALL_stacked_key_dist_BASELINE_decayed, HUMAN_stacked_key_dist, HUMAN_stacked_key_presses, occlusion_frames = None):

    if occlusion_frames is None:
        keypress_dist_over_time_model = jnp.concatenate(list(ALL_stacked_key_dist.values()), axis=0)
        keypress_dist_over_time_frozen = jnp.concatenate(list(ALL_stacked_key_dist_BASELINE_frozen.values()), axis=0)
        keypress_dist_over_time_decayed = jnp.concatenate(list(ALL_stacked_key_dist_BASELINE_decayed.values()), axis=0)
        keypress_dist_over_time_HUMAN = jnp.concatenate(list(HUMAN_stacked_key_dist.values()), axis=0)
        ALL_human_bools = jnp.concatenate([jnp.sum(h_key_presses != 2, axis = 0) for h_key_presses in HUMAN_stacked_key_presses.values()], axis = 0)
    else:
        keypress_dist_over_time_model = jnp.concatenate([v[jnp.array(occlusion_frames[k])] for k,v in ALL_stacked_key_dist.items()], axis = 0)
        keypress_dist_over_time_frozen = jnp.concatenate([v[jnp.array(occlusion_frames[k])] for k,v in ALL_stacked_key_dist_BASELINE_frozen.items()], axis = 0)
        keypress_dist_over_time_decayed = jnp.concatenate([v[jnp.array(occlusion_frames[k])] for k,v in ALL_stacked_key_dist_BASELINE_decayed.items()], axis = 0)
        keypress_dist_over_time_HUMAN = jnp.concatenate([v[jnp.array(occlusion_frames[k])] for k,v in HUMAN_stacked_key_dist.items() if k in ALL_stacked_key_dist], axis = 0)
        ALL_human_bools = jnp.concatenate([jnp.sum(h_key_presses != 2, axis = 0)[jnp.array(occlusion_frames[k])] for k,h_key_presses in HUMAN_stacked_key_presses.items() if k in ALL_stacked_key_dist], axis = 0)

    # P(Green or Red)
    DECISION_DIST_PRESSED_BUTTON_model = jnp.sum(keypress_dist_over_time_model[:,:2], axis = 1)
    DECISION_DIST_PRESSED_BUTTON_frozen = jnp.sum(keypress_dist_over_time_frozen[:,:2], axis = 1)
    DECISION_DIST_PRESSED_BUTTON_decayed = jnp.sum(keypress_dist_over_time_decayed[:,:2], axis = 1)
    DECISION_DIST_PRESSED_BUTTON_HUMAN = jnp.sum(keypress_dist_over_time_HUMAN[:,:2], axis = 1)

    # P(Green | Green or Red)
    DECISION_DIST_CONDITIONAL_GREEN_model = keypress_dist_over_time_model[:,0]/DECISION_DIST_PRESSED_BUTTON_model
    DECISION_DIST_CONDITIONAL_GREEN_frozen = keypress_dist_over_time_frozen[:,0]/DECISION_DIST_PRESSED_BUTTON_frozen
    DECISION_DIST_CONDITIONAL_GREEN_decayed = keypress_dist_over_time_decayed[:,0]/DECISION_DIST_PRESSED_BUTTON_decayed
    DECISION_DIST_CONDITIONAL_GREEN_HUMAN = keypress_dist_over_time_HUMAN[:,0]/DECISION_DIST_PRESSED_BUTTON_HUMAN

    common_time_points_press_model = jnp.logical_not(jnp.logical_or(check_invalid(DECISION_DIST_CONDITIONAL_GREEN_model), check_invalid(DECISION_DIST_CONDITIONAL_GREEN_HUMAN)))
    common_time_points_press_frozen = jnp.logical_not(jnp.logical_or(check_invalid(DECISION_DIST_CONDITIONAL_GREEN_frozen), check_invalid(DECISION_DIST_CONDITIONAL_GREEN_HUMAN)))
    common_time_points_press_decayed = jnp.logical_not(jnp.logical_or(check_invalid(DECISION_DIST_CONDITIONAL_GREEN_decayed), check_invalid(DECISION_DIST_CONDITIONAL_GREEN_HUMAN)))
                                                    
    valid_model_conditional_green = DECISION_DIST_CONDITIONAL_GREEN_model[common_time_points_press_model]
    valid_human_conditional_green_v_model = DECISION_DIST_CONDITIONAL_GREEN_HUMAN[common_time_points_press_model]
    correl_weights_model = ALL_human_bools[common_time_points_press_model]
    weighted_model_green_conditional_pearsonr = weighted_corr(valid_model_conditional_green, valid_human_conditional_green_v_model, correl_weights_model)
    model_conditional_green_pearsonr = pearsonr(valid_model_conditional_green, valid_human_conditional_green_v_model)[0]                                               

    valid_frozen_conditional_green = DECISION_DIST_CONDITIONAL_GREEN_frozen[common_time_points_press_frozen]
    valid_human_conditional_green_v_frozen = DECISION_DIST_CONDITIONAL_GREEN_HUMAN[common_time_points_press_frozen]
    correl_weights_frozen = ALL_human_bools[common_time_points_press_frozen]
    weighted_frozen_green_conditional_pearsonr = weighted_corr(valid_frozen_conditional_green, valid_human_conditional_green_v_frozen, correl_weights_frozen)
    frozen_conditional_green_pearsonr = pearsonr(valid_frozen_conditional_green, valid_human_conditional_green_v_frozen)[0]

    valid_decayed_conditional_green = DECISION_DIST_CONDITIONAL_GREEN_decayed[common_time_points_press_decayed]
    valid_human_conditional_green_v_decayed = DECISION_DIST_CONDITIONAL_GREEN_HUMAN[common_time_points_press_decayed]
    correl_weights_decayed = ALL_human_bools[common_time_points_press_decayed]
    weighted_decayed_green_conditional_pearsonr = weighted_corr(valid_decayed_conditional_green, valid_human_conditional_green_v_decayed, correl_weights_decayed)
    decayed_conditional_green_pearsonr = pearsonr(valid_decayed_conditional_green, valid_human_conditional_green_v_decayed)[0]

    return keypress_dist_over_time_model, keypress_dist_over_time_frozen, keypress_dist_over_time_decayed, keypress_dist_over_time_HUMAN, \
        DECISION_DIST_PRESSED_BUTTON_model, DECISION_DIST_PRESSED_BUTTON_frozen, DECISION_DIST_PRESSED_BUTTON_decayed, DECISION_DIST_PRESSED_BUTTON_HUMAN, \
        model_conditional_green_pearsonr, frozen_conditional_green_pearsonr, decayed_conditional_green_pearsonr, \
        weighted_model_green_conditional_pearsonr, weighted_frozen_green_conditional_pearsonr, weighted_decayed_green_conditional_pearsonr, \
        correl_weights_model, correl_weights_frozen, correl_weights_decayed, \
        valid_model_conditional_green, valid_frozen_conditional_green, valid_decayed_conditional_green, \
        valid_human_conditional_green_v_model, valid_human_conditional_green_v_frozen, valid_human_conditional_green_v_decayed


def create_log_frequency_heatmaps_model_only(
    human_pressed_button, model_pressed_button,
    human_choose_green_v_model,
    model_choose_green, 
    correl_weights_model=None,
    bins=20, weighted=False, cmap='viridis', cmap_reverse=False, model_name='JTAP'
):
    """
    Create 6 log-frequency heatmaps arranged in a 3x2 grid comparing human and model data.
    
    Args:
        human_pressed_button, model_pressed_button, frozen_pressed_button, decayed_pressed_button:
            Arrays for P(Red or Green).
        human_choose_green_*, *_choose_green: Arrays for P(Green | Red or Green).
        bins: Number of bins for the heatmap (default: 20).
        weighted: Whether to use correlation weights (default: False).
        cmap: Matplotlib colormap name (default: 'viridis').
        cmap_reverse: Whether to reverse the colormap (default: False).
    
    Returns:
        matplotlib.figure.Figure: The complete figure with all heatmaps.
    """
    # Set style
    plt.style.use('default')
    sns.set_style("white")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(1, 2, figure=fig, hspace=0.7, wspace=0.15)
    
    # Define bin edges
    bin_edges = np.linspace(0, 1, bins + 1)
    
    def calculate_heatmap(human_data, model_data, correl_weights):
        # Digitize both arrays into bins

        human_bin_idx = np.digitize(human_data, bin_edges) - 1
        human_bin_idx[human_bin_idx == bins] = bins - 1
        model_bin_idx = np.digitize(model_data, bin_edges) - 1
        model_bin_idx[model_bin_idx == bins] = bins - 1
        
        # Initialize the 2D histogram
        heatmap = np.zeros((bins, bins))
        heatmap_weight = np.zeros((bins, bins))
        
        # Accumulate counts
        for h, m, weight in zip(human_bin_idx, model_bin_idx, correl_weights):
            heatmap[h, m] += 1
            heatmap_weight[h, m] += weight
            
        # Normalize weights
        if np.sum(heatmap_weight) > 0:
            heatmap_weight /= np.sum(heatmap_weight)
            
        # Compute log frequencies
        log_heatmap = np.log1p(heatmap) + np.log1p(heatmap_weight)
        return log_heatmap
    
    # Generate all heatmaps
    uniform_weights = np.zeros_like(human_pressed_button)
    heatmaps_pressed = [
        calculate_heatmap(human_pressed_button, model_pressed_button, uniform_weights),
    ]

    pressed_correl = [
        pearsonr(human_pressed_button, model_pressed_button)[0],
    ]
    
    correl_weights = [
        np.zeros_like(human_choose_green_v_model) if (correl_weights_model is None or not weighted) else correl_weights_model,
    ]
    
    heatmaps_green = [
        calculate_heatmap(human_choose_green_v_model, model_choose_green, correl_weights[0]),
    ]

    green_correl = [
        weighted_corr(human_choose_green_v_model, model_choose_green,correl_weights_model),
    ]
    
    # Set up colormaps
    if cmap_reverse:
        cmap = plt.get_cmap(cmap).reversed()
    else:
        cmap = plt.get_cmap(cmap)
    
    # Determine global color scale
    vmin = min(np.min(hp) for hp in heatmaps_pressed + heatmaps_green)
    vmax = max(np.max(hp) for hp in heatmaps_pressed + heatmaps_green)
    
    # Titles and labels
    # column_headers = ['P(Red or Green)', 'P(Green | Red or Green)']
    column_headers = ['$\\mathbf{P(Decision)}$', '$\\mathbf{P(Green \\; | \\; Decision)}$']

    # Plot heatmaps
    for row, (hp, hg) in enumerate(zip(heatmaps_pressed, heatmaps_green)):
        # Left heatmap (P(Red or Green))
        ax1 = fig.add_subplot(gs[row, 0])
        im1 = ax1.imshow(hp, cmap=cmap, origin='lower', extent=[0, 1, 0, 1], 
                        aspect='equal', vmin=vmin, vmax=vmax)
        # ax1.set_title(title, pad=15, fontsize=12, fontweight='bold')
        
        # Right heatmap (P(Green | Red or Green))
        ax2 = fig.add_subplot(gs[row, 1])
        im2 = ax2.imshow(hg, cmap=cmap, origin='lower', extent=[0, 1, 0, 1], 
                        aspect='equal', vmin=vmin, vmax=vmax)
        
        ax1.set_xlabel(model_name, fontsize=24, fontweight='bold', labelpad=15)
        ax2.set_xlabel(model_name, fontsize=24, fontweight='bold', labelpad=15)

        ax1.set_ylabel('Participants', fontsize=24, fontweight='bold', labelpad=15)

        # ax1.set_title(f"$r = {pressed_correl[row]:.2f}$", fontsize=22, fontweight='bold')
        # ax2.set_title(f"$r_{{\\text{{wtd}}}} = {green_correl[row]:.2f}$", fontsize=22, fontweight='bold')
        ax1.text(0.01, 0.925, f"$r = {pressed_correl[row]:.2f}$", fontsize=24, fontweight='bold', transform=ax1.transAxes)
        ax2.text(0.01, 0.925, f"$r_{{\\text{{wtd}}}} = {green_correl[row]:.2f}$", fontsize=24, fontweight='bold', transform=ax2.transAxes)
        ax1.set_title(column_headers[0], fontsize=24, fontweight='bold', pad=20)
        ax2.set_title(column_headers[1], fontsize=24, fontweight='bold', pad = 20)

        # Grid settings
        for ax in [ax1, ax2]:
            ax.grid(False)
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.set_yticks(np.linspace(0, 1, 5))
            # Add spines styling
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('#333333')
            
            # Improve tick appearance
            ax.tick_params(width=1.5, length=6, color='#333333', labelsize = 14)

        # fig.text(0.25 + row * 0.5, 1 - (row * 0.3), title, ha='center', va='center', 
        #         fontsize=14, fontweight='bold')
        
        # After plotting all heatmaps, add a single colorbar at the bottom
    cbar_ax = fig.add_axes([0.35, 0.025, 0.3, 0.04])  # Adjust position as needed
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='Log Frequency')
    cbar.set_label('Log Frequency', fontsize=20)  # Set the label font size explicitly
    cbar.ax.tick_params(labelsize=14)

    # Add column headers
    # fig.text(0.3, 0.91, column_headers[0], ha='center', va='center', 
    #          fontsize=22, fontweight='bold')
    # fig.text(0.72, 0.91, column_headers[1], ha='center', va='center', 
    #          fontsize=22, fontweight='bold')

    return fig