import jax.numpy as jnp
import jax
from .initialization import is_target_fully_hidden, is_target_intersecting_rectangle
from .scene_geometry import edge_intersection_time_double_vmap, wrap_corner_intersection_time_circle_vmap, generate_target_stacked_edges
from jax.debug import print as jprint

# @jax.jit
def maybe_resolve_collision(size, x,y, vx, vy, edgemap, cornermap, dist_to_travel):

    speed = jnp.sqrt(vx**2 + vy**2)
    time_left = jnp.round(dist_to_travel/speed,3)

    overlap = 0.50 - 1e-3
    edge_collision_data = edge_intersection_time_double_vmap(
        generate_target_stacked_edges(x,y, size), edgemap, vx, vy, overlap
    )

    t_edges = edge_collision_data[...,0].flatten()
    t_edges_horizontal = jnp.where(edge_collision_data[...,1].flatten(), t_edges, jnp.float32(jnp.inf))
    t_edges_vertical = jnp.where(edge_collision_data[...,2].flatten(), t_edges, jnp.float32(jnp.inf))

    # return edges
    min_edge_time_horizontal = t_edges_horizontal.min()
    edge_time_horizontal = jnp.where(
        min_edge_time_horizontal <= time_left,
        min_edge_time_horizontal, jnp.float32(jnp.inf)
    )
    min_edge_time_vertical = t_edges_vertical.min()
    edge_time_vertical = jnp.where(
        min_edge_time_vertical <= time_left,
        min_edge_time_vertical, jnp.float32(jnp.inf)
    )


    min_corner_time, vx2, vy2, c_times = wrap_corner_intersection_time_circle_vmap(
        x, y, size, cornermap, vx, vy,
    )

    corner_time = jnp.where(
        min_corner_time <= time_left,
        min_corner_time, jnp.float32(jnp.inf)
    )

    result = jnp.select(
        [
            jnp.isfinite(edge_time_horizontal) & (edge_time_horizontal == edge_time_vertical) & (edge_time_horizontal <= corner_time),
            jnp.isfinite(edge_time_horizontal) & (edge_time_horizontal < edge_time_vertical) & (edge_time_horizontal <= corner_time),
            jnp.isfinite(edge_time_vertical) & (edge_time_vertical <= corner_time),
            jnp.isfinite(corner_time)
        ],
        jnp.array([
            [
                1,  # x_collision
                1,  # y_collision
                jnp.round(dist_to_travel - edge_time_horizontal * speed, 3),  # remaining_dist
                x + vx * edge_time_horizontal,  # new_x
                y + vy * edge_time_horizontal,  # new_y
                vx * -1,  # new_vx
                vy * -1,  # new_vy
                0.,  # collision_branch
            ],
            # Horizontal edge collision
            [
                0,  # x_collision
                1,  # y_collision
                jnp.round(dist_to_travel - edge_time_horizontal * speed, 3),  # remaining_dist
                x + vx * edge_time_horizontal,  # new_x
                y + vy * edge_time_horizontal,  # new_y
                vx * 1,  # new_vx
                vy * -1,  # new_vy
                1.,  # collision_branch
            ],
            # Vertical edge collision
            [
                1,  # x_collision
                0,  # y_collision
                jnp.round(dist_to_travel - edge_time_vertical * speed, 3),  # remaining_dist
                x + vx * edge_time_vertical,  # new_x
                y + vy * edge_time_vertical,  # new_y
                vx * -1,  # new_vx
                vy * 1,  # new_vy
                2.,  # collision_branch
            ],
            # Corner collision
            [
                1,  # x_collision
                1,  # y_collision
                jnp.round(dist_to_travel - corner_time * speed, 3),  # remaining_dist
                x + vx * corner_time,  # new_x
                y + vy * corner_time,  # new_y
                vx2,  # new_vx
                vy2,  # new_vy
                3.,  # collision_branch
            ]
        ], dtype= jnp.float32),
        jnp.array([
            0,  # x_collision
            0,  # y_collision
            0.,  # remaining_dist
            x + vx * time_left,  # new_x
            y + vy * time_left,  # new_y
            vx,  # new_vx
            vy,  # new_vy
            4.,  # collision_branch
        ], dtype= jnp.float32)
    )

    return result, c_times

def velocity_transform(mo, mi, friction, elasticity, size, inference_mode_bool):

    shape = mo.shape
    speed, dir = mo.speed, mo.dir
    vx, vy = speed * jnp.cos(dir), speed * jnp.sin(dir)
    dist_to_travel = jnp.sqrt(vx**2 + vy**2)
    next_x, next_y, next_vx, next_vy = mo.x,mo.y, vx, vy
    edgemap = mo.edgemap
    cornermap = mo.cornermap

    def loop_condition(carry):
        *_, dist_to_travel, col_iter = carry
        physics_cond = jnp.logical_and(jnp.logical_not(jnp.isclose(dist_to_travel, jnp.float32(0), atol=1e-03)), jnp.greater_equal(dist_to_travel,jnp.float32(0)))
        early_stop_cond = jnp.less(col_iter, mi.max_num_col_iters)
        return jnp.logical_and(physics_cond, early_stop_cond)

    def loop_body(carry):
        size, collision_detect_x, collision_detect_y, old_x, old_y, old_vx, old_vy, collision_branch, edgemap, cornermap, dist_to_travel, col_iter = carry
        (collision_detect_x, collision_detect_y, dist_to_travel, \
            next_x, next_y, next_vx, next_vy, new_collision_branch), c_times = maybe_resolve_collision(size, old_x, old_y, old_vx, old_vy, edgemap, cornermap, dist_to_travel)
        collision_branch = jnp.where(jnp.equal(new_collision_branch, jnp.float32(4.)), collision_branch, new_collision_branch)

        return (size, collision_detect_x, collision_detect_y, next_x, next_y, next_vx, next_vy, collision_branch, edgemap, cornermap, dist_to_travel, col_iter+jnp.float32(1))

    # init collision branch is a dummy -1
    initial_carry = (size, jnp.float32(0),jnp.float32(0), next_x, next_y, next_vx, next_vy, jnp.float32(4), edgemap, cornermap, dist_to_travel, jnp.float32(0))
    # Execute the while loop
    _, collision_detect_x, collision_detect_y, next_x, next_y, next_vx, next_vy, collision_branch, *_, col_iter = jax.lax.while_loop(loop_condition, loop_body, initial_carry)

    stopped_early = jnp.greater_equal(col_iter, mi.max_num_col_iters)


    is_target_hidden = is_target_fully_hidden(next_x, next_y, size, mo.masked_occluders)
    is_target_visible = jnp.logical_not(is_target_intersecting_rectangle(next_x, next_y, size, mo.masked_occluders))
    is_target_partially_hidden = jnp.logical_not(jnp.logical_or(is_target_hidden, is_target_visible))

    next_speed = speed

    speed_noise = jnp.where(inference_mode_bool, mi.σ_speed, mi.σ_speed_sim)

    next_dir = jnp.arctan2(next_vy, next_vx)
    σ_COL_dir = jnp.where(inference_mode_bool, mi.σ_COL_dir, mi.σ_COL_dir_sim)
    σ_NOCOL_dir = jnp.where(inference_mode_bool, mi.σ_NOCOL_dir, mi.σ_NOCOL_dir_sim)

    dir_σ = jnp.where(collision_branch == jnp.float32(4), σ_NOCOL_dir, σ_COL_dir)
        
    return next_speed, speed_noise, next_dir, dir_σ, next_x, next_y, \
        collision_branch, is_target_hidden, is_target_visible, \
        is_target_partially_hidden, stopped_early