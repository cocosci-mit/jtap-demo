from genjax import gen
from .stepper import stepper_model
from .initialization import init_model
from .likelihood import *

def get_render_args(mi, mo):
   return (mi.pix_x, mi.pix_y, mo.shape, mo.size,
                     mo.x, mo.y, mo.masked_barriers, mo.masked_occluders, 
                     mo.red_sensor, mo.green_sensor)

@gen
def full_init_model(mi):
   init_mo = init_model(mi) @ "init"
   likelihood_model(get_render_args(mi, init_mo), mi.flip_prob, mi.filter_size_arr, mi.σ_pix_blur) @ "init_obs"
   return init_mo

@gen
def full_step_model(mo, mi, inference_mode_bool):
   mo = stepper_model(mo, mi, inference_mode_bool) @ "step"
   likelihood_model(get_render_args(mi,mo), mi.flip_prob, mi.filter_size_arr, mi.σ_pix_blur) @ "obs"
   return mo
