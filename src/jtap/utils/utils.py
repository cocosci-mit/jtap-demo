import os
import json
import jax
import genjax
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu

def get_package_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def get_assets_dir():
    return os.path.join(get_package_dir(), "assets")

def get_fonts_dir():
    return os.path.join(get_assets_dir(), "fonts")

# asserts during import
assert os.path.exists(get_package_dir())
assert os.path.exists(get_fonts_dir())


def to_serializable(val):
    if isinstance(val, np.ndarray):
        return {'type': 'numpy', 'data': val.tolist()}  # Mark as numpy array
    elif isinstance(val, jnp.ndarray):
        return {'type': 'jax', 'data': val.tolist()}  # Mark as jax.numpy array
    elif isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [to_serializable(item) for item in val]
    return val
    
def from_serializable(val):
    if isinstance(val, dict):
        if 'type' in val and 'data' in val:
            # Check if it's marked as a numpy or jax.numpy array
            if val['type'] == 'numpy':
                return np.array(val['data'])
            elif val['type'] == 'jax':
                return jnp.array(val['data'])
        return {k: from_serializable(v) for k, v in val.items()}
    elif isinstance(val, list):
        # No need to check for numeric types specifically now, handle recursively
        return [from_serializable(item) for item in val]
    return val


def determine_sensor_output(sample):
    if 'x' in sample:
        x = sample['x']
        y = sample['y']
    else:
        x,y = sample['xy']

    size = sample['size']

    in_red = not (
        (x+size <= sample['red_x']) or
        (y >= sample['red_y'] + sample['red_size_y']) or
        (x >= sample['red_x'] + sample['red_size_x']) or
        (y+size <= sample['red_y'])
    )

    in_green = not (
        (x+size <= sample['green_x']) or
        (y >= sample['green_y'] + sample['green_size_y']) or
        (x >= sample['green_x'] + sample['green_size_x']) or
        (y+size <= sample['green_y'])
    )

    if ((in_green and in_red) or ((not in_green) and (not in_red))):
        return None
    elif in_red:
        return 'r'
    else:
        return 'g'        


def determine_sensor_output_full_model(sample, red_sensor, green_sensor, size):
    x = sample['x'].value
    y = sample['y'].value
    # size = sample['size'].value

    red_x, red_y, red_size_x, red_size_y = red_sensor
    green_x, green_y, green_size_x, green_size_y = green_sensor
   
    in_red = not (
        (x+size <= red_x) or
        (y >= red_y + red_size_y) or
        (x >= red_x + red_size_x) or
        (y+size <= red_y)
    )

    in_green = not (
        (x+size <= green_x) or
        (y >= green_y + green_size_y) or
        (x >= green_x + green_size_x) or
        (y+size <= green_y)
    )

    if ((in_green and in_red) or ((not in_green) and (not in_red))):
        return None
    elif in_red:
        return 'r'
    else:
        return 'g'        

def get_observations(tr, obs_smoothing_scale = 1):
   skip_t = int(1./obs_smoothing_scale)
   init_obs_array = tr.get_sample()['init_obs']
   step_obs_arrays = tr.get_sample()['all_steps',..., 'obs'].value[skip_t-1::skip_t]
   return jnp.concatenate([init_obs_array[None,...], step_obs_arrays])

find_nearest = jax.jit(lambda A, target : A[jnp.argmin(jnp.abs(A - target))])
find_nearest_valid = jax.jit(lambda A, target, valid : A[jnp.argmin(jnp.abs(jnp.where(valid,A,jnp.inf) - target))])

#######################
# Model/Trace-related utils #
#######################

get_step = lambda p: p.get_subtrace(('all_steps',)).inner.inner.inner.get_subtrace(('step',))


###############
# Pytree utils
###############
    
def concat_pytrees(pytree_list, axis = 0):
    return jtu.tree_map(lambda *xs: jnp.concatenate(xs, axis = axis), *pytree_list)

def stack_pytrees(pytree_list, axis = 0):
    return jtu.tree_map(lambda *ys: jnp.stack(ys, axis = axis), *pytree_list)

# JAX pytree leaves only
def reshape_first_dim_pytree(pytree, shape):
    def reshape(x):
        new_shape = (*shape,*x.shape[1:])
        return x.reshape(new_shape)
    return jtu.tree_map(reshape, pytree)

def split_pytree(pytree, num_splits):
    return [jtu.tree_map(lambda x: jnp.array_split(x, num_splits)[i], pytree) for i in range(num_splits)]

def flatten_pytree(pytree):
    return jtu.tree_map(lambda x: x.flatten(), pytree)

def swap_axes_pytree(pytree, axis1, axis2):
    def swap_axes(x):
        if hasattr(x, 'shape') and len(x.shape) > max(axis1, axis2):
            return jnp.swapaxes(x, axis1, axis2)
        return x  # Leave non-array or insufficiently dimensional objects unchanged
    
    return jax.tree_util.tree_map(swap_axes, pytree)

def slice_pytree(pytree, i):
    return jtu.tree_map(lambda v : v[i], pytree)

slice_pt = slice_pytree

def safe_slice_pytree(pytree, i, unsafe_dim_len = jnp.inf):
    return jtu.tree_map(
        lambda v: v[i] if hasattr(v, 'shape') and 
        len(v.shape) > 0 and v.shape[0] > i 
        and v.shape[0] != unsafe_dim_len
        else v, 
        pytree
    )

def safe_fromend_slice_pytree(pytree, endoffset, unsafe_dim_len = jnp.inf):
    return jtu.tree_map(
        lambda v: v[:-endoffset] if hasattr(v, 'shape') and 
        len(v.shape) > 0 and v.shape[0] > 0 
        and v.shape[0] != unsafe_dim_len
        else v, 
        pytree
    )

def init_step_concat(init, steps):
    return jtu.tree_map(
        lambda a, b: jnp.concatenate([a[None, ...], b], axis=0), 
        init, steps
    )

def multislice_pytree(pytree, indices):
    return jax.vmap(slice_pytree, in_axes = (None, 0))(pytree, indices)


def flattened_multivmap(f, unflatten_output = True):
    """
    This flattened VMAP does a multi-vmap 
    over a flattened collapsed representation
    along with memory-saving array splits
    """
    def _get_vmapped(f, n):
        f = jax.jit(jax.vmap(f, in_axes = tuple(0 for _ in range(n))))
        return f

    def inner(map_encoding, *args):
        # NOTE THAT LAST DIMENSION MUST BE MULTIMAPPED FOR NOW
        assert map_encoding[-1] == 1
        # in map_encoding, 1 means multivmap and 0 means single vmap over flattened dimension
        n = len(args)
        assert len(map_encoding) == n
        vmapped_fn = _get_vmapped(f, n)

        multivmap_args = tuple(x for i,x in enumerate(args) if map_encoding[i] == 1)
        singlevmap_args = tuple(x for i,x in enumerate(args) if map_encoding[i] == 0)

        shapes = tuple([x.shape[0] for x in multivmap_args])
        # print(shapes)
        extended_args = jnp.meshgrid(*multivmap_args, indexing = 'ij')
        extended_args_flattened = []
        multi_count = 0
        for i in range(n):
            if map_encoding[i] == 1:
                extended_args_flattened.append(extended_args[multi_count].flatten())
                multi_count += 1
            else:
                extended_args_flattened.append(singlevmap_args[i - multi_count])
        extended_args_flattened = tuple(extended_args_flattened)
        # print([x.shape for x in extended_args_flattened])
        # extended_args_flattened = tuple(x.flatten() for x in extended_args)
        len_arg = extended_args_flattened[-1].shape[0]
        num_segs = max(len_arg//1000, 1)
        # print(len_arg, num_segs)
        extended_args_flattened_split = tuple(split_pytree(x,num_segs) for x in extended_args_flattened)
        split_outputs = []
        for i in range(num_segs):
            # print(f"seg {i+1} of {num_segs}")
            split_outputs.append(vmapped_fn(*tuple(x[i] for x in extended_args_flattened_split)))
        
        combined_output = concat_pytrees(split_outputs)
        
        if unflatten_output:
            return reshape_first_dim_pytree(combined_output, shapes)
        else:
            return combined_output
    
    return inner

def multivmap(f, num_vmap_dims = None):
    """
    Given a function `f` of `n` arguments, return a function
    on `n` jax vectors of arguments `a1, .., an`, which outputs an n-dimensional
    array `A` such that `A[i1, ..., in] = f(a1[i1], ..., an[in])`.
    """
    def _get_multivmapped(f, n, num_vmap_dims):
        if num_vmap_dims is None:
            num_vmap_dims = n
        for i in range(num_vmap_dims - 1, -1, -1):
            f = jax.vmap(f, in_axes=tuple(0 if j == i else None for j in range(n)))
        return f

    def inner(*args):
        n = len(args)
        return _get_multivmapped(f, n, num_vmap_dims)(*args)
    
    return inner

# general utils
d2r = jax.jit(lambda x: jnp.float32(x) * jnp.pi / 180)
r2d = jax.jit(lambda x: jnp.float32(x) * 180 / jnp.pi)

# 2d int packing from int8 to int4

@jax.jit
def pack_int4_2d(values):
    """
    Pack a 2D array of int4 values into a 2D array of int8.
    Each pair of int4 values along the last dimension is packed into one int8.
    """
    assert values.shape[-1] % 2 == 0, "Number of columns must be even to pack int4 values."
    values = jnp.clip(values, -8, 7).astype(jnp.int8)  # Simulate signed int4

    # Split the array into high and low parts
    high = values[:, 0::2]  # Take every even column
    low = values[:, 1::2]   # Take every odd column

    # Pack the int4 values
    packed = (high << 4) | (low & 0xF)  # Combine high and low parts
    return packed

@jax.jit
def unpack_int4_2d(packed):
    """
    Unpack a 2D array of int8 values into a 2D array of int4.
    Each int8 is unpacked into two int4 values along the last dimension.
    """
    # Extract high and low parts
    high = (packed >> 4) & 0xF
    low = packed & 0xF

    # Convert to signed int4 range
    high = jnp.where(high >= 8, high - 16, high)
    low = jnp.where(low >= 8, low - 16, low)

    # Combine high and low into a 2D array
    unpacked = jnp.empty((packed.shape[0], packed.shape[1] * 2), dtype=jnp.int8)
    unpacked = unpacked.at[:, 0::2].set(high)  # Set every even column
    unpacked = unpacked.at[:, 1::2].set(low)   # Set every odd column
    return unpacked
