import chex
import jax.numpy as jnp
import numpy as np
from collections import namedtuple
import dataclasses
from typing import Tuple, List, Dict


@chex.dataclass(frozen=True)
class ChexModelInput:
    scene_dim: Tuple[int, int]
    σ_pos: float
    reappearing_σ_pos : float
    σ_size: float
    σ_speed: float
    reappearing_σ_speed : float
    σ_friction: float   
    σ_elasticity: float
    σ_NOCOL_dir: float
    σ_COL_dir: float
    reappearing_dir_σ: float
    σ_barrier: float
    σ_barrier_size: float
    σ_occ: float
    σ_occ_size: float
    σ_sensor_size: float
    σ_sensor: float
    flip_prob: float
    filter_size: int
    σ_pix_blur: float
    max_speed: float
    max_friction: float
    min_elasticity: float
    max_barrier_size: float
    max_num_barriers: float
    max_occ_size: float
    max_num_occ: float
    max_sensor_size: float
    image_discretization: float
    T : float
    num_x_grid : int
    num_y_grid : int
    grid_size_x : float
    grid_size_y : float
    size_lims : jnp.ndarray
    max_num_col_iters : int
    num_x_grid_arr : jnp.ndarray = None
    num_y_grid_arr : jnp.ndarray = None
    pix_x: jnp.ndarray = None
    pix_y: jnp.ndarray = None
    filter_size_arr: jnp.ndarray = None
    barrier_num_probs: jnp.ndarray = None
    occ_num_probs: jnp.ndarray = None
    empty_edgemap : Dict = None
    empty_cornermap : Dict = None
    T_arr : jnp.ndarray = None
    shape_idxs : jnp.ndarray = None
    σ_pos_sim: float = None
    σ_size_sim: float = None
    σ_speed_sim: float = None
    σ_NOCOL_dir_sim: float = None
    σ_COL_dir_sim: float = None
    σ_speed_occ: float = None
    σ_NOCOL_dir_occ: float = None
    σ_COL_dir_occ: float = None
    σ_pos_initprop: float = None
    σ_size_initprop: float = None
    σ_speed_initprop: float = None
    σ_NOCOL_dir_initprop: float = None
    σ_NOCOL_dir_stepprop: float = None
    σ_COL_dir_prop : float = None
    σ_speed_stepprop: float = None
    σ_pos_stepprop: float = None

    def prepare_input(self):
        object.__setattr__(self, "pix_x", jnp.array(np.arange(0, self.scene_dim[0], self.image_discretization)).astype(jnp.float32))
        object.__setattr__(self, "pix_y", jnp.array(np.arange(0, self.scene_dim[1], self.image_discretization)).astype(jnp.float32))
        object.__setattr__(self, "barrier_num_probs", jnp.ones(int(self.max_num_barriers + 1))/(self.max_num_barriers + 1))
        object.__setattr__(self, "occ_num_probs", jnp.ones(int(self.max_num_occ + 1))/(self.max_num_occ + 1))
        object.__setattr__(self, "T_arr", jnp.zeros(self.T))
        object.__setattr__(self, "num_x_grid_arr", jnp.zeros(self.num_x_grid))
        object.__setattr__(self, "num_y_grid_arr", jnp.zeros(self.num_y_grid))
        object.__setattr__(self, "filter_size_arr", jnp.zeros(self.filter_size))
        object.__setattr__(self, "empty_edgemap", self.prepare_empty_edgemap(int(self.max_num_barriers)))
        object.__setattr__(self, "empty_cornermap", self.prepare_empty_cornermap(int(self.max_num_barriers)))
        # depend on inference params
        if self.σ_pos_sim is None:
            object.__setattr__(self, "σ_pos_sim", self.σ_pos)
        if self.σ_size_sim is None:
            object.__setattr__(self, "σ_size_sim", self.σ_size)
        if self.σ_speed_sim is None:
            object.__setattr__(self, "σ_speed_sim", self.σ_speed)
        if self.σ_NOCOL_dir_sim is None:
            object.__setattr__(self, "σ_NOCOL_dir_sim", self.σ_NOCOL_dir)
        if self.σ_COL_dir_sim is None:
            object.__setattr__(self, "σ_COL_dir_sim", self.σ_COL_dir)
        # depend on sim_params
        if self.σ_speed_occ is None:
            object.__setattr__(self, "σ_speed_occ", self.σ_speed_sim)
        if self.σ_NOCOL_dir_occ is None:
            object.__setattr__(self, "σ_NOCOL_dir_occ", self.σ_NOCOL_dir_sim)
        if self.σ_COL_dir_occ is None:
            object.__setattr__(self, "σ_COL_dir_occ", self.σ_COL_dir_sim)
        # depend on simulation params
        if self.σ_pos_initprop is None:
            object.__setattr__(self, "σ_pos_initprop", self.σ_pos_sim)
        if self.σ_size_initprop is None:
            object.__setattr__(self, "σ_size_initprop", self.σ_size_sim)
        if self.σ_speed_initprop is None:
            object.__setattr__(self, "σ_speed_initprop", self.σ_speed_sim)
        if self.σ_NOCOL_dir_initprop is None:
            object.__setattr__(self, "σ_NOCOL_dir_initprop", self.σ_NOCOL_dir_sim)
        # depend on initprops
        if self.σ_NOCOL_dir_stepprop is None:
            object.__setattr__(self, "σ_NOCOL_dir_stepprop", self.σ_NOCOL_dir_initprop)
        if self.σ_COL_dir_prop is None:
            object.__setattr__(self, "σ_COL_dir_prop", self.σ_COL_dir_sim)
        if self.σ_speed_stepprop is None:
            object.__setattr__(self, "σ_speed_stepprop", self.σ_speed_initprop)
        if self.σ_pos_stepprop is None:
            object.__setattr__(self, "σ_pos_stepprop", self.σ_size_initprop)

    def update(self, attr, value):
        object.__setattr__(self, attr, value)

    def prepare_empty_edgemap(self, max_num_barriers):
        # Total edges = 4 wall edges + 4 edges per barrier
        max_num_edges = 4 + max_num_barriers * 4
        return {
            "stacked_edges": jnp.zeros((max_num_edges, 2, 2), dtype=jnp.float32),
            "valid": jnp.zeros(max_num_edges, dtype=jnp.bool_),
        }

    def prepare_empty_cornermap(self, max_num_barriers):
        # 4 wall corners + 4 * num of barriers
        max_num_corners = 4 + max_num_barriers * 4
        return {
            "stacked_corners": jnp.zeros((max_num_corners, 4), dtype=jnp.float32),
            "valid": jnp.zeros(max_num_corners, dtype=jnp.bool_),
        }
    
    def generate_input(self):
        # Prepare input values
        self.prepare_input()
        # Create a namedtuple with all fields
        ModelInput = namedtuple('ModelInput', [field.name for field in dataclasses.fields(self)])
        # Return an instance of the namedtuple
        return ModelInput(*[getattr(self, field.name) for field in dataclasses.fields(self)])

ModelOutput = namedtuple("ModelOutput", [
    "shape", "size", "x", "y", "speed", "dir", "num_occs", "num_barriers",
    "masked_occluders", "masked_barriers", "is_target_hidden",
    "is_target_partially_hidden", "is_target_visible", "is_reappearing",
    "red_sensor", "green_sensor", "collision_branch", "last_collision_data",
    "elasticity", "friction", "edgemap", "cornermap", "T", "stopped_early"
])
