import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Arrow, Circle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from jtap.utils import *
import jtap
import numpy as np
import jax.numpy as jnp
from IPython.display import HTML as HTML_Display
import jax
import genjax
from tqdm import tqdm
from PIL import Image
from jax.scipy.special import logsumexp
import pandas as pd
from jtap.inference import get_rg_expectation

def generate_samples(key, n_samples, support, logprobs):
    keys = jax.random.split(key, n_samples)
    sampled_idxs = jax.vmap(jax.random.categorical, in_axes = (0, None))(keys, logprobs)
    return support[sampled_idxs]

generate_samples_vmap = jax.vmap(generate_samples, in_axes = (0,None,0,0))

def red_green_viz(JTAP_data, obs_arrays, prediction_t_offset = 5, video_offset = (0,0),
    fps = 10, skip_t = 1, show_latents = True, inference_input = None, viz_key = None, 
    min_dot_alpha = 0.2, min_line_alpha = 0.04, show_resampled_text = True, num_t_steps = None):
    ###
    """
    ASSUMPTIONS: 
    1. SCENE IS STATIC (no changing barriers & occluders)
    2. View is overlayed on observations
    """
    ###

    max_line_alpha = 1 # should always be 1, makes no sense to have alpha > 1
    min_line_alpha = min_line_alpha

    max_dot_alpha = 1 # should always be 1, makes no sense to have alpha > 1
    min_dot_alpha = min_dot_alpha

    if num_t_steps is None:
        num_inference_steps = JTAP_data['weights'].shape[0]
    else:
        num_inference_steps = num_t_steps
    num_prediction_steps = JTAP_data['JTAP_params']['max_prediction_steps']
    max_inference_T = num_inference_steps - 1
    maxt = obs_arrays.shape[0]
    n_particles = JTAP_data['JTAP_params']['num_particles']
    normalized_weights_ = jnp.exp(JTAP_data['weights'] - logsumexp(JTAP_data['weights'], axis = 1, keepdims = True)) # T by N_PARTICLES

    equal_prob_value = 1/n_particles

    particle_collapsed = jnp.any(jnp.isnan(normalized_weights_), axis = 1)

    normalized_weights = jnp.where(jnp.isnan(normalized_weights_), jnp.full_like(normalized_weights_, equal_prob_value), normalized_weights_)

    if max_inference_T > maxt:
        print("Too many timesteps in particle data")
        return

    if num_inference_steps - video_offset[0] - video_offset[1] <= 0:
        print(f"Video limits are too restrictive. Plotting from T = {video_offset[0]}"+\
              f" to T = {max_inference_T - video_offset[1]} " +\
              "is not possible")
        return

    if num_prediction_steps < prediction_t_offset:
        print(f"Prediction offset is too high. Prediction offset is {prediction_t_offset}"+\
              f" but max prediction steps is {num_prediction_steps}")
        return
    
    max_inference_T_for_video = max_inference_T - video_offset[1]


    inf_x_points = JTAP_data['tracking']['x'] # T by N_PARTICLES
    inf_y_points = JTAP_data['tracking']['y'] # T by N_PARTICLES
    pred_x_lines = jnp.concatenate([JTAP_data['tracking']['x'][:,None,:], JTAP_data['prediction']['x']], axis = 1) # T by T_pred+1 by N_PARTICLES
    pred_y_lines = jnp.concatenate([JTAP_data['tracking']['y'][:,None,:], JTAP_data['prediction']['y']], axis = 1) # T by T_pred+1 by N_PARTICLES
    inf_dots_alpha_over_time = min_dot_alpha + normalized_weights * (max_dot_alpha - min_dot_alpha)
    pred_alphas_over_time = min_line_alpha + normalized_weights * (max_line_alpha - min_line_alpha)
    sizes_over_time = JTAP_data['tracking']['size'] # T by N_PARTICLES


    color_mapping = {
        0: (255, 255, 255),  # white
        1: (128, 128, 128),  # grey
        2: (0, 0, 255),      # blue
        3: (0, 0, 0),        # black
        4: (255, 0, 0),      # red
        5: (0, 255, 0)       # green
    }
    # Create a list of the RGB colors in the order of their keys
    color_list = [color_mapping[i] for i in range(6)]
    color_array = np.array(color_list, dtype=np.uint8)
    frames = []
    for arr in obs_arrays:
        rgb_array = color_array[np.rot90(np.array(arr, dtype = np.uint8),k=1)]
        image = Image.fromarray(rgb_array)
        frames.append(np.array(image))
        
    # Create a figure
    if show_latents:
        fig = plt.figure(figsize=(10,8))
        gs = GridSpec(12, 12, figure=fig)
        ax1 = fig.add_subplot(gs[:7, :7])
        ax2 = fig.add_subplot(gs[1:6, 8:]) 
        ax3 = fig.add_subplot(gs[8:, :7], projection = 'polar')
        ax4 = fig.add_subplot(gs[8:, 8:])  
    else:
        fig = plt.figure(figsize=(10,5))
        gs = GridSpec(1, 12, figure=fig)
        ax1 = fig.add_subplot(gs[0, :6])
        ax2 = fig.add_subplot(gs[0, 7:]) 

    ax1.set_aspect('equal', 'box')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_yticks([])
    ax1.set_title('Position (' + r'$S_x, S_y$' +')')

    # AX1

    # WORKS WELL NOW
    scale_multiplier = 1 / JTAP_data["JTAP_params"]["image_discretization"]
    x_to_pix_x = jax.jit(lambda x, s : (x + 0.5*s) * scale_multiplier)
    y_to_pix_y = jax.jit(lambda y, s : image.height - 1 - (y + 0.5*s) * scale_multiplier)

    x_to_pix_x_vmap = jax.vmap(x_to_pix_x)
    y_to_pix_y_vmap = jax.vmap(y_to_pix_y)

    filtering_posterior_dot_probs = ax1.scatter(x_to_pix_x_vmap(inf_x_points[0], sizes_over_time[0]), 
        y_to_pix_y_vmap(inf_y_points[0], sizes_over_time[0]),
        s = 20,
        c = 'k', linewidths = 0,
        zorder=5,
        alpha = inf_dots_alpha_over_time[0])
            
    p_lines = []
    for n in range(n_particles):
        p_lines.append(Line2D(x_to_pix_x(pred_x_lines[0,:,n], sizes_over_time[0,n]), 
            y_to_pix_y(pred_y_lines[0,:,n], sizes_over_time[0,n]), color='orange', 
            alpha=round(float(pred_alphas_over_time[0,n]),2), zorder=4, linestyle="-"))
        ax1.add_line(p_lines[n])
        
    im = ax1.imshow(frames[0])

    timer_text = fig.text(0.1, 0.95, "Timestep: " + r'$\bf{0}$', ha='left', color="k", fontsize=15)
    if show_resampled_text:
        resampled_text = fig.text(0.4, 0.95, f"Resampled: {JTAP_data['resampled'][0]}", ha='left', color="b", fontsize=17)

    particle_collapsed_text = fig.text(0.7, 0.95, f"Particle Collapsed: {particle_collapsed[0]}", ha='left', color="r", fontsize=15)

    # AX2
    rg_data = get_rg_expectation(JTAP_data, prediction_t_offset)
    bars = ax2.bar(range(3), rg_data[0], color = ['green', 'red', 'blue'])

    # Set up the axis limits
    # Set title and x-ticks
    ax2.set_title('Red or Green: Which will it hit next?', fontsize=10)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['Green', 'Red', 'Uncertain'])
    ax2.set_ylim(0, 1)

    if show_latents:
        # AX3
        num_bins = 90
        n_samples = 10000
        max_count = 10
        viz_key, dir_key = jax.random.split(viz_key, 2)
        dir_keys = jax.random.split(dir_key, num_inference_steps)
        sampled_dir = generate_samples_vmap(dir_keys, 10000, JTAP_data['tracking']['dir'][:num_inference_steps], JTAP_data['weights'][:num_inference_steps])
        all_counts_dir = []
        # NOTE: IN THIS VIZ, STEP DIR IS TAKEN  a step before current step
        for i in range(max_inference_T_for_video + 1):
            counts_dir, bin_edges_dir = np.histogram(sampled_dir[i], bins=num_bins, range=(-np.pi, np.pi), density=True)
            # rescale to sqrt of counts for it to be compatible with volume of pie histogram
            counts_dir = np.sqrt(counts_dir)
            all_counts_dir.append(counts_dir * (max_count / max(counts_dir)))

        bin_centers = (bin_edges_dir[:-1] + bin_edges_dir[1:]) / 2
        
        # Plot the histogram on the polar plot
        dir_bars = ax3.bar(bin_centers, all_counts_dir[0], width=bin_edges_dir[1] - bin_edges_dir[0], bottom=0, color='brown', edgecolor=None, alpha=0.6)
        ax3.grid(False)
        ax3.set_title('Direction (' + r'$\phi$' +')')
        ax3.set_ylim(0, max_count)
        ax3.set_yticklabels([])
        ax3.set_theta_zero_location("E")
        ax3.set_theta_direction(1)
        ax3.set_xticklabels(['0°', '45°', '90°', '135°', '±180°', '-135°', '-90°', '-45°'])

       # AX4
        num_bins = int(inference_input.max_speed/ 0.1) + 1
        n_samples = 10000
        max_count = 10
        viz_key, speed_key = jax.random.split(viz_key, 2)
        speed_keys = jax.random.split(speed_key, num_inference_steps)
        sampled_speed = generate_samples_vmap(speed_keys, 10000, JTAP_data['tracking']['speed'][:num_inference_steps], JTAP_data['weights'][:num_inference_steps])
        all_counts_speed = []
        # NOTE: IN THIS VIZ, STEP SPEED IS TAKEN  a step before current step
        for i in range(max_inference_T_for_video + 1):
            counts_speed, bin_edges_speed = np.histogram(sampled_speed[i], bins=num_bins, range=(0, inference_input.max_speed), density=True)
            all_counts_speed.append(counts_speed * (max_count / sum(counts_speed)))

        bin_centers = (bin_edges_speed[:-1] + bin_edges_speed[1:]) / 2
        speed_bars = ax4.bar(bin_centers, all_counts_speed[0], width=bin_edges_speed[1] - bin_edges_speed[0], bottom=0, color='orange', edgecolor=None, alpha=0.6)
        # ax4.grid(False)
        ax4.set_title('Speed (' + r'$\nu$' +')')
        ax4.set_ylim(0, max_count)
        ax4.set_yticks([])    
        ax4.set_yticklabels([])

    def init_func():
        pass

    def update(frame_idx):
        # if idx <= video_offset[0]:
        #     return
        frame_idx += skip_t*(video_offset[0])
        im.set_array(frames[frame_idx])
        if frame_idx % skip_t == 0:
            idx = int(frame_idx/skip_t)
            timer_text.set_text(f"Timestep: " + r'$\bf{' + str(idx) + r'}$')
            particle_collapsed_text.set_text(f"Particle Collapsed: {particle_collapsed[idx]}")
            if show_resampled_text:
                if particle_collapsed[idx]:
                    resampled_text.set_color("r")
                    resampled_text.set_text("Resampled: Disabled")
                else:
                    if JTAP_data['resampled'][idx]:
                        resampled_text.set_color("g")
                    else:
                        resampled_text.set_color("b")
                    resampled_text.set_text(f"Resampled: {JTAP_data['resampled'][idx]}")
            print(f"rendering inference step {idx}")
            filtering_posterior_dot_probs.set_offsets(np.c_[
                x_to_pix_x_vmap(inf_x_points[idx], 
                    sizes_over_time[idx]),
                y_to_pix_y_vmap(inf_y_points[idx], 
                    sizes_over_time[idx])
            ])
            filtering_posterior_dot_probs.set_alpha(inf_dots_alpha_over_time[idx])
            for n in range(n_particles):

                p_lines[n].set_data(x_to_pix_x(pred_x_lines[idx,:,n], sizes_over_time[idx,n]), 
                    y_to_pix_y(pred_y_lines[idx,:,n], sizes_over_time[idx,n]))
                p_lines[n].set_alpha(round(float(pred_alphas_over_time[idx,n]),2))

            for bar, height in zip(bars, rg_data[idx]):
                bar.set_height(height)
            if show_latents:
                for bar, height in zip(dir_bars, all_counts_dir[idx]):
                    bar.set_height(height)
                for bar, height in zip(speed_bars, all_counts_speed[idx]):
                    bar.set_height(height)

    n_frames = max_inference_T_for_video + 1 - video_offset[0]
    animation = FuncAnimation(fig, update, frames=skip_t*n_frames, init_func=init_func, interval=1000//fps)
    plt.close()
    return HTML_Display(animation.to_html5_video())