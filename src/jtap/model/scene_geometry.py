import jax
import jax.numpy as jnp

@jax.jit
def generate_wall_edges(scene_dim, edgemap):
    right, top = scene_dim

    wall_edges = jnp.array([
        [[jnp.float32(0), jnp.float32(0)], [jnp.float32(0), top]], # left
        [[jnp.float32(0), top], [right, top]], # top
        [[right, jnp.float32(0)], [right, top]], # right
        [[jnp.float32(0), jnp.float32(0)], [right, jnp.float32(0)]] # bottom
    ])

    edgemap["stacked_edges"] = edgemap["stacked_edges"].at[:4].set(wall_edges)
    edgemap["valid"] = edgemap["valid"].at[:4].set(jnp.ones(4, dtype=jnp.bool_))
    return edgemap

@jax.jit
def get_valid_barrier_edges(barrier):
    x, y, size_x, size_y = barrier
    left, right, top, bottom = x, x + size_x, y + size_y, y
    
    edges = jnp.array([
        [[left, bottom], [left, top]],
        [[left, top], [right, top]],
        [[right, bottom], [right, top]],
        [[left, bottom], [right, bottom]]
    ], dtype=jnp.float32)
    return edges, jnp.ones(4, dtype=jnp.bool_)

@jax.jit
def get_invalid_barrier_edges(_):
    return jnp.zeros((4, 2, 2), dtype=jnp.float32), jnp.zeros(4, dtype=jnp.bool_)

@jax.jit
def generate_barrier_edges(edgemap, masks):
    def get_barrier_edges(mask):
        return jax.lax.cond(
            mask.flag, get_valid_barrier_edges, get_invalid_barrier_edges, mask.value
        )

    stacked_edges, valid_flags = jax.vmap(get_barrier_edges)(masks)
    stacked_edges = stacked_edges.reshape(-1, 2, 2)
    valid_flags = valid_flags.flatten()

    edgemap_size = edgemap["valid"].shape[0]
    edgemap["stacked_edges"] = edgemap["stacked_edges"].at[4:edgemap_size].set(stacked_edges)
    edgemap["valid"] = edgemap["valid"].at[4:edgemap_size].set(valid_flags)
    return edgemap

@jax.jit
def get_edges_from_scene(scene_dim, edgemap, masked_barriers):
    edgemap = generate_wall_edges(scene_dim, edgemap)
    return generate_barrier_edges(edgemap, masked_barriers)

@jax.jit
def generate_target_stacked_edges(x, y, size):
    return jnp.array([
        # Bottom edge
        [x, y, x + size, y, 
                   jnp.float32(1.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)],
        # Left edge
        [x, y, x, y + size, 
                   jnp.float32(0.0), jnp.float32(1.0), jnp.float32(0.0), jnp.float32(0.0)],
        # Right edge
        [x + size, y, x + size, y + size, 
                   jnp.float32(0.0), jnp.float32(0.0), jnp.float32(1.0), jnp.float32(0.0)],
        # Top edge
        [x, y + size, x + size, y + size, 
                   jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(1.0)]
    ])


def check_same_height_or_breadth_overlap(x1,x2,x3,x4, overlap = jnp.float32(0.0)):
    return jnp.logical_and(jnp.greater(x2-x3,overlap), jnp.greater(x4-x1, overlap))

def edge_intersection_time(targetedge, edgemap, vx, vy, overlap):

    no_collision_retval = jnp.array([
        jnp.float32(jnp.inf), jnp.float32(0), jnp.float32(0)]
    )

    def inner_():
        x1, y1 = targetedge[:2]
        x2, y2 = targetedge[2:4]
        bottom, left, right, top = targetedge[4:]
        x3, y3 = edgemap["stacked_edges"][0]
        x4, y4 = edgemap["stacked_edges"][1]

        # jax.debug.print(
        #     "Edge Points: (x1={x1}, y1={y1}), (x2={x2}, y2={y2}), (x3={x3}, y3={y3}), (x4={x4}, y4={y4})",
        #     x1=jnp.asarray(x1), y1=jnp.asarray(y1), 
        #     x2=jnp.asarray(x2), y2=jnp.asarray(y2), 
        #     x3=jnp.asarray(x3), y3=jnp.asarray(y3), 
        #     x4=jnp.asarray(x4), y4=jnp.asarray(y4)
        # )

        targetedge_width = x2-x1
        targetedge_height = y2-y1

        is_horizontal_case = (y1 == y2) & (y3 == y4)
        is_vertical_case = (y1 != y2) & (y3 != y4)
        
        # Handling all cases with JAX's control flow
        def horizontal_case():
            return jax.lax.cond(
                jnp.equal(vy,jnp.float32(0)),
                lambda : no_collision_retval,
                lambda : jax.lax.cond(
                    jnp.logical_and(
                        check_same_height_or_breadth_overlap(
                            jnp.round(x1 + (vx/vy)*(y3-y1), 3), 
                            jnp.round(targetedge_width + x1 + (vx/vy)*(y3-y1),3), 
                            jnp.round(x3,3), jnp.round(x4,3),
                            overlap
                        ),
                        jnp.logical_and(
                            jnp.logical_or(
                                jnp.logical_and(
                                    jnp.less(vy,jnp.float32(0)), bottom
                                ),
                                jnp.logical_and(
                                    jnp.greater(vy,jnp.float32(0)), top
                                )
                            ),
                            jnp.logical_or(
                                jnp.equal(jnp.sign(jnp.round(y3 - y1,3)),jnp.sign(jnp.round(vy,3))),
                                jnp.equal(jnp.round(y3 - y1,3), jnp.float32(0))
                            )
                        )
                    ),
                    lambda : jnp.array([jnp.round((y3 - y1)/ vy,3), jnp.float32(1), jnp.float32(0)]),
                    lambda : no_collision_retval
                )
            )

        
        def vertical_case():
            return jax.lax.cond(
                jnp.equal(vx,jnp.float32(0)),
                lambda : no_collision_retval,
                lambda : jax.lax.cond(
                    jnp.logical_and(
                        check_same_height_or_breadth_overlap(
                            jnp.round(y1 + (vy/vx)*(x3-x1),3), 
                            jnp.round(targetedge_height + y1 + (vy/vx)*(x3-x1),3), 
                            jnp.round(y3,3), jnp.round(y4,3),
                            overlap
                        ),
                        jnp.logical_and(
                            jnp.logical_or(
                                jnp.logical_and(
                                    jnp.less(vx,jnp.float32(0)), left
                                ),
                                jnp.logical_and(
                                    jnp.greater(vx,jnp.float32(0)), right
                                )
                            ),
                            jnp.logical_or( 
                                jnp.equal(jnp.sign(jnp.round(x3 - x1,3)),jnp.sign(jnp.round(vx,3))),
                                jnp.equal(jnp.round(x3 - x1,3), jnp.float32(0))
                            )
                        )
                    ),
                    lambda : jnp.array([jnp.round((x3 - x1)/ vx,3), jnp.float32(0), jnp.float32(1)]),
                    lambda : no_collision_retval
                )
            )

        return jax.lax.cond(
            is_horizontal_case,
            horizontal_case,
            lambda: jax.lax.cond(
                is_vertical_case,
                vertical_case,
                lambda: no_collision_retval
            )
        )

    return jax.lax.cond(edgemap['valid'], inner_, lambda: no_collision_retval)
                                                
    
edge_intersection_time_double_vmap = jax.vmap(
    jax.vmap(
        edge_intersection_time, 
        in_axes=(None, 0, None, None, None)
    ), in_axes=(0, None, None, None, None)
)

@jax.jit
def generate_wall_corners(scene_dim, cornermap):
    right, top = scene_dim

    wall_corners = jnp.array([
        [0.0, 0.0, 1, 1],    # BL
        [right, 0.0, -1, 1],  # BR
        [0.0, top, 1, -1],   # TL
        [right, top, -1, -1] # TR
    ], dtype=jnp.float32)

    cornermap["stacked_corners"] = cornermap["stacked_corners"].at[:4].set(wall_corners)
    cornermap["valid"] = cornermap["valid"].at[:4].set(jnp.ones(4, dtype=jnp.bool_))
    return cornermap

@jax.jit
def get_valid_barrier_corners(barrier):
    x, y, size_x, size_y = barrier
    left, right, top, bottom = x, x + size_x, y + size_y, y
    
    corners = jnp.array([
        [left, bottom, 1, 1],   # BL
        [right, bottom, -1, 1], # BR
        [left, top, 1, -1],    # TL
        [right, top, -1, -1]   # TR
    ], dtype=jnp.float32)
    
    return corners, jnp.ones(4, dtype=jnp.bool_)

@jax.jit
def get_invalid_barrier_corners(_):
    return jnp.zeros((4, 4), dtype=jnp.float32), jnp.zeros(4, dtype=jnp.bool_)

@jax.jit
def generate_barrier_corners(cornermap, masks):
    def get_barrier_corners(mask):
        return jax.lax.cond(
            mask.flag, get_valid_barrier_corners, get_invalid_barrier_corners, mask.value
        )

    stacked_corners, valid_flags = jax.vmap(get_barrier_corners)(masks)
    stacked_corners = stacked_corners.reshape(-1, 4)
    valid_flags = valid_flags.flatten()

    cornermap_size = cornermap["valid"].shape[0]
    cornermap["stacked_corners"] = cornermap["stacked_corners"].at[4:cornermap_size].set(stacked_corners)
    cornermap["valid"] = cornermap["valid"].at[4:cornermap_size].set(valid_flags)
    return cornermap

@jax.jit
def remove_duplicate_corners(cornermap):
    stacked_corners = cornermap["stacked_corners"]
    valid = cornermap["valid"]

    def outer_(corner_i):
        def inner_(corner_j):
            return jax.lax.cond(
                jnp.all(jnp.array([
                    valid[corner_i],
                    valid[corner_j],
                    jnp.all(stacked_corners[corner_i][:2] == stacked_corners[corner_j][:2]),
                    jnp.any(jnp.array([
                        jnp.all(stacked_corners[corner_i][2:] == -stacked_corners[corner_j][2:]),
                    ]))
                ])),
                lambda: 1,
                lambda: 0
            )

        return jnp.sum(jax.vmap(inner_)(jnp.arange(stacked_corners.shape[0])))

    duplicate_arr = jax.vmap(outer_)(jnp.arange(stacked_corners.shape[0]))
    cornermap["valid"] = jnp.where(duplicate_arr > 0, False, cornermap["valid"])
    return cornermap

@jax.jit
def get_corners_from_scene(scene_dim, cornermap, masked_barriers):
    cornermap = generate_wall_corners(scene_dim, cornermap)
    cornermap = generate_barrier_corners(cornermap, masked_barriers)
    cornermap = remove_duplicate_corners(cornermap)
    return cornermap

@jax.jit
def generate_target_stacked_corners(x, y, size):
    # Define the four target corners directly as JAX arrays
    corners = jnp.array([
        [x, y, 1, 1],               # Bottom-left (BL)
        [x + size, y, -1, 1],      # Bottom-right (BR)
        [x, y + size, 1, -1],      # Top-left (TL)
        [x + size, y + size, -1, -1]  # Top-right (TR)
    ], dtype=jnp.float32)
    
    return corners


@jax.jit
def reflect_velocity(vx, vy, cx, cy, x1, y1):
    # Calculate the direction vector of the line AB
    dx = x1 - cx
    dy = y1 - cy
    
    # Calculate the magnitude of the direction vector
    mag = jnp.round(jnp.sqrt(dx**2 + dy**2),3)
    
    # Normalize the direction vector to get the unit direction vector
    ux = jnp.round(dx / mag, 3)
    uy = jnp.round(dy / mag, 3)
    
    # Find the normal vector to the line AB
    nx = -uy
    ny = ux
    
    # Calculate the dot product of the velocity vector and the normal vector
    dot_product = vx * nx + vy * ny
    
    # Calculate the reflected velocity components
    vx2 = vx - jnp.float32(2) * dot_product * nx
    vy2 = vy - jnp.float32(2) * dot_product * ny
    
    return jnp.float32(-1)*vx2, jnp.float32(-1)*vy2

@jax.jit
def corner_intersection_time_circle_inner(A, B, discriminant, cx, cy, x_center, y_center, vx, vy):
    # just check x-axis coordinate to get the time if vx =/= 0, else check with vy
    t1 = jnp.round((-B + jnp.sqrt(discriminant))/(jnp.float32(2)*A),3)
    t2 = jnp.round((-B - jnp.sqrt(discriminant))/(jnp.float32(2)*A),3)

    time_taken_1 = jnp.where(jnp.logical_and(jnp.greater(t1, 0), jnp.less_equal(t1, 1)), t1, jnp.float32(jnp.inf))
    time_taken_2 = jnp.where(jnp.logical_and(jnp.greater(t2, 0), jnp.less_equal(t2, 1)), t2, jnp.float32(jnp.inf))

    min_t = jnp.min(jnp.array([time_taken_1, time_taken_2]))
    x1_new = min_t*vx + x_center
    y1_new = min_t*vy + y_center

    vx2, vy2 = reflect_velocity(vx, vy, cx, cy, x1_new, y1_new)

    return min_t, vx2, vy2

@jax.jit
def corner_intersection_time_circle(r, x, y, cornermap, vx, vy):
    # assume radius = 0.5
    # formula from https://mathworld.wolfram.com/Circle-LineIntersection.html
    # https://chatgpt.com/share/66fb6b4d-9134-8004-848a-f1f2c319394f
    def inner():
        cx, cy = cornermap["stacked_corners"][0], cornermap["stacked_corners"][1]
        x1, y1, x2, y2 = x - cx + r,y - cy + r, x + vx - cx + r, y + vy - cy + r # assume corner centered at 0, 0
        dx = x2 - x1
        dy = y2 - y1
        A = jnp.square(dx) + jnp.square(dy)
        B = jnp.float32(2)*(x1*dx + y1*dy)
        C = jnp.square(x1) + jnp.square(y1) - jnp.square(r)
        discriminant = jnp.round(jnp.square(B) - jnp.float32(4)*A*C,3)
        # NOTE: positive discriminant means that velocity
        # vector intersects, does not mean that it is in the correct direction

        # handling discrete jumps
        gap_2 = jnp.square(x1) + jnp.square(y1)
        # peturb object in velocity (less than radial distance) to see if it is moving towards barrier
        coefs = jnp.arange(0.1,1.01,0.01)
        speed = jnp.sqrt(jnp.square(vx) + jnp.square(vy))
        dts = coefs *jnp.min(jnp.array([r,speed]))/speed
        gaps_after_2 = jnp.square(x1 + dts*vx) + jnp.square(y1+dts*vy)
        moving_towards_barrier = jnp.all(jnp.less(jnp.round(gaps_after_2,3), jnp.round(gap_2, 3)))
        penetration = jnp.less(jnp.round(gap_2, 3), jnp.round(jnp.square(r),3))

        return jax.lax.cond(
            penetration,
            lambda: jax.lax.cond(
                moving_towards_barrier,
                lambda: (jnp.float32(0.), *reflect_velocity(vx, vy, cx, cy, x+r, y+r)),
                lambda: (jnp.float32(jnp.inf), jnp.float32(jnp.inf), jnp.float32(jnp.inf))
            ),
            lambda: jax.lax.cond(
                jnp.logical_and(
                    jnp.greater(discriminant, jnp.float32(0)),
                    moving_towards_barrier
                ),
                corner_intersection_time_circle_inner,
                lambda *_ : (jnp.float32(jnp.inf), jnp.float32(jnp.inf), jnp.float32(jnp.inf)),
                A, B, discriminant, cx, cy, x+r, y+r, vx, vy
            )
        )

    return jax.lax.cond(cornermap["valid"], inner, lambda: (jnp.float32(jnp.inf), jnp.float32(jnp.inf), jnp.float32(jnp.inf)))

corner_intersection_time_circle_vmap = jax.vmap(
        corner_intersection_time_circle,
 in_axes=(None, None, None, 0, None, None)
)

@jax.jit
def wrap_corner_intersection_time_circle_vmap(x, y, size, cornermap, vx, vy):
    radius = jnp.float32(0.5) * size
    corner_times, vx2s, vy2s = corner_intersection_time_circle_vmap(radius, x, y, cornermap, vx, vy)
    min_idx = corner_times.argmin()
    return corner_times[min_idx], vx2s[min_idx], vy2s[min_idx], corner_times