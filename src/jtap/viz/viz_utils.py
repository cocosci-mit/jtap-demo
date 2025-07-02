import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import vonmises
from IPython.display import HTML as HTML_Display
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os


def display_video(frames, framerate=30, skip_t = 1):
    """
    frames: list of N np.arrays (H x W x 3)
    framerate: frames per second
    """
    height, width, _ = frames[0].shape
    dpi = 70

    num_frames = len(frames)
    if skip_t != 1:
      frames = frames[::skip_t]
    # orig_backend = matplotlib.get_backend()
    # matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    max_figsize_width = 6

    fig, ax = plt.subplots(1, 1, figsize=(max_figsize_width, max_figsize_width*(height/width)))
    # matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    # ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frames[frame])
        # return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=np.arange(frames.shape[0]),
      interval=interval, blit=False, repeat=True)
    plt.close()
    return HTML_Display(anim.to_html5_video())