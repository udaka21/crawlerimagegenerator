import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle

# --- Robot parameters ---
BODY_LEN = 2.0
BODY_H   = 0.35
PIVOT_Y  = 0.35                 # body center height above origin
FRONT_PIVOT_X = +0.95          # pivot x in BODY frame (forward +)
REAR_PIVOT_X  = -0.95          # pivot x in BODY frame (backward -)

FLIPPER_LEN = 1.0
FLIPPER_THK = 0.15

# --- Joint appearance/placement ---
JOINT_R      = 0.08            # joint circle radius
JOINT_OFFSET = 0.06            # above body bottom edge (in BODY frame)

def rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def draw_flipper_rect(ax, pivot_world, angle_world_rad, length, thickness, color='0.75'):
    """Draw a thick rectangular flipper given a world-frame pivot and angle."""
    rect_local = np.array([
        [0,       -thickness/2],
        [length,  -thickness/2],
        [length,   thickness/2],
        [0,        thickness/2]
    ])
    R = rot2d(angle_world_rad)
    rect_world = (R @ rect_local.T).T + pivot_world
    ax.add_patch(Polygon(rect_world, closed=True, facecolor=color, edgecolor='k'))
    return rect_world

def draw_robot(ax, body_angle_deg=0, front_rel_deg=0, rear_rel_deg=0):
    """
    Draw robot and return (xmin, xmax, ymin, ymax).
    Angles:
      - body_angle_deg: body rotation in WORLD frame
      - front_rel_deg:  front flipper angle relative to body x-axis (forward)
      - rear_rel_deg:   rear flipper angle relative to body x-axis but pointing backward
                        (we use world angle = body + (180° - rear_rel))
    """
    # Body center in WORLD frame
    body_center = np.array([0.0, PIVOT_Y])
    body_th = np.deg2rad(body_angle_deg)

    # --- Body rectangle in BODY frame (centered at origin) ---
    halfL, halfH = BODY_LEN/2, BODY_H/2
    body_local = np.array([
        [-halfL,  halfH],
        [ halfL,  halfH],
        [ halfL, -halfH],
        [-halfL, -halfH]
    ])
    # rotate to WORLD and translate
    body_world = (rot2d(body_th) @ body_local.T).T + body_center
    body_patch = Polygon(body_world, closed=True, facecolor='lightblue', edgecolor='k', zorder=1)
    ax.add_patch(body_patch)

    # --- Pivot (joint) positions in BODY frame (then rotate+translate to WORLD) ---
    y_pivot_local = -halfH + JOINT_OFFSET
    front_pivot_local = np.array([FRONT_PIVOT_X, y_pivot_local])
    rear_pivot_local  = np.array([REAR_PIVOT_X,  y_pivot_local])

    front_pivot = (rot2d(body_th) @ front_pivot_local) + body_center
    rear_pivot  = (rot2d(body_th) @ rear_pivot_local)  + body_center

    # --- Flippers (angles converted to WORLD) ---
    front_world_th = body_th + np.deg2rad(front_rel_deg)
    # rear: point backward, then apply -rear_rel around body x-axis
    rear_world_th  = body_th + (np.pi - np.deg2rad(rear_rel_deg))

    fl1 = draw_flipper_rect(ax, front_pivot, front_world_th, FLIPPER_LEN, FLIPPER_THK, color='0.75')
    fl2 = draw_flipper_rect(ax, rear_pivot,  rear_world_th,  FLIPPER_LEN, FLIPPER_THK, color='0.75')

    # --- Joints (small circles) ---
    ax.add_patch(Circle(front_pivot, JOINT_R, facecolor='white', edgecolor='k', zorder=3))
    ax.add_patch(Circle(rear_pivot,  JOINT_R, facecolor='white', edgecolor='k', zorder=3))

    # --- Compute tight bounding box of everything we drew ---
    all_pts = np.vstack([body_world, fl1, fl2,
                         front_pivot[None,:], rear_pivot[None,:]])
    pad = 0.08  # small visual margin
    xmin, ymin = all_pts.min(axis=0) - pad
    xmax, ymax = all_pts.max(axis=0) + pad

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.0, 2.0)

    return xmin, xmax, ymin, ymax

# ---- Example: body=10°, front=+5° (relative to body), rear=−53° (relative to body) ----
if __name__ == "__main__":
    body_angle = 0
    front_angle = 20
    rear_angle  = 20

    fig, ax = plt.subplots(figsize=(5, 3))
    xmin, xmax, ymin, ymax = draw_robot(ax, body_angle, front_angle, rear_angle)

    # # Size figure to content (optional; improves raster resolution control)
    # width, height = (xmax-xmin), (ymax-ymin)
    # fig.set_size_inches(width*2.5, height*2.5)  # scale factor for DPI

    filename = f"robot_b{body_angle}_f{front_angle}_r{rear_angle}.png"
    plt.savefig(filename, dpi=400, transparent=True, bbox_inches='tight', pad_inches=0)
    print("Saved:", filename)
    plt.show()