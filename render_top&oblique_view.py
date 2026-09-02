import argparse
import json
import os

import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import pyrender
import trimesh
from PIL import Image

from utils.topview import render_topview_zup_no_margin


def render_4_sideviews(o3d_mesh, out_dir="sideviews", width=1024, height=1024):
    """Render RGB + depth images from 4 horizontal directions around the mesh."""
    os.makedirs(out_dir, exist_ok=True)

    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    if not o3d_mesh.has_vertex_colors():
        raise ValueError("Mesh has no vertex colors.")

    colors = np.asarray(o3d_mesh.vertex_colors)
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors,
        process=False
    )

    # Bounding box
    bounds = mesh.bounds
    xmin, ymin, zmin = bounds[0]
    xmax, ymax, zmax = bounds[1]

    z_top = zmax
    z_mid = (zmin + zmax) / 2
    x_mid = (xmin + xmax) / 2
    y_mid = (ymin + ymax) / 2

    # 4 camera positions (midpoints of top edges)
    cam_positions = [
        np.array([x_mid, ymin, z_top + 1.0]),  # front edge
        np.array([xmax, y_mid, z_top + 1.0]),  # right edge
        np.array([x_mid, ymax, z_top + 1.0]),  # back edge
        np.array([xmin, y_mid, z_top + 1.0]),  # left edge
    ]

    # Center of opposite vertical faces
    targets = [
        np.array([x_mid, ymax, z_mid]),  # opposite of front -> back face center
        np.array([xmin, y_mid, z_mid]),  # opposite of right -> left face
        np.array([x_mid, ymin, z_mid]),  # opposite of back -> front face
        np.array([xmax, y_mid, z_mid]),  # opposite of left -> right face
    ]

    scene = pyrender.Scene(
        bg_color=[255, 255, 255, 255],
        ambient_light=[1.0, 1.0, 1.0]
    )

    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(render_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(90.0))

    renderer = pyrender.OffscreenRenderer(width, height)

    for i in range(4):
        cam_pos = cam_positions[i]
        target = targets[i]

        forward = target - cam_pos
        forward /= np.linalg.norm(forward)

        up = np.array([0, 0, 1])  # Z-up
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = cam_pos

        cam_node = scene.add(camera, pose=pose)

        color, depth = renderer.render(
            scene,
            flags=pyrender.RenderFlags.FLAT
        )

        Image.fromarray(color).save(
            os.path.join(out_dir, f"view_{i + 1}_rgb.png")
        )

        depth_mask = depth > 0
        depth_norm = np.zeros_like(depth)
        if depth_mask.any():
            d_min = depth[depth_mask].min()
            d_max = depth[depth_mask].max()
            depth_norm[depth_mask] = (depth[depth_mask] - d_min) / (d_max - d_min + 1e-8)

        depth_color = cm.viridis(depth_norm)[:, :, :3]
        depth_color[~depth_mask] = [1, 1, 1]  # no geometry -> white
        depth_color = (depth_color * 255).astype(np.uint8)

        Image.fromarray(depth_color).save(
            os.path.join(out_dir, f"view_{i + 1}_depth.png")
        )

        scene.remove_node(cam_node)

    renderer.delete()

    return out_dir


def render_local_views(pruned_mesh, original_mesh, out_dir="local_views", width=512, height=512):
    """
    Render 2 local RGB side-view images of the pruned scene (MCDR step, Sec. 3.3).

    For each axis (x, y) of the pruned bounding box, two candidate camera positions
    are considered. The one closer to the original scene center is selected so the
    camera faces outward (away from the scene interior), reducing look-through-wall
    artefacts. The camera is slightly elevated above the box center to reduce
    occlusion by surrounding furniture. Depth maps are discarded.

    Output: local_views/local_x.png, local_views/local_y.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # Convert pruned o3d mesh to trimesh for pyrender
    vertices = np.asarray(pruned_mesh.vertices)
    faces = np.asarray(pruned_mesh.triangles)
    colors = np.asarray(pruned_mesh.vertex_colors)
    if colors.size > 0 and colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)

    # Pruned bounding box
    bounds = tri_mesh.bounds
    xmin, ymin, zmin = bounds[0]
    xmax, ymax, zmax = bounds[1]
    xc, yc, zc = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    xe, ye, ze = xmax - xmin, ymax - ymin, zmax - zmin

    # Original scene center (for outward-facing camera selection)
    orig_aabb = original_mesh.get_axis_aligned_bounding_box()
    orig_min = np.asarray(orig_aabb.get_min_bound())
    orig_max = np.asarray(orig_aabb.get_max_bound())
    scene_cx = (orig_min[0] + orig_max[0]) / 2
    scene_cy = (orig_min[1] + orig_max[1]) / 2

    # Slight elevation above box center (half the pruned height)
    cam_z = zc + ze * 0.5
    target = np.array([xc, yc, zc])

    # pyrender scene
    render_scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[1.0, 1.0, 1.0])
    render_scene.add(pyrender.Mesh.from_trimesh(tri_mesh, smooth=False))
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(90.0))
    renderer = pyrender.OffscreenRenderer(width, height)

    out_paths = []

    # Camera distance: enough to see the full face of the box perpendicular to that axis
    dx = max(ye, ze)   # looking along x → need to see y-z face
    dy = max(xe, ze)   # looking along y → need to see x-z face

    axes = [
        ("x", "local_x.png",
         [np.array([xmax + dx, yc, cam_z]),   # candidate from +x
          np.array([xmin - dx, yc, cam_z])],  # candidate from -x
         lambda c: abs(c[0] - scene_cx)),
        ("y", "local_y.png",
         [np.array([xc, ymax + dy, cam_z]),   # candidate from +y
          np.array([xc, ymin - dy, cam_z])],  # candidate from -y
         lambda c: abs(c[1] - scene_cy)),
    ]

    for _axis, fname, candidates, dist_fn in axes:
        cam_pos = candidates[int(np.argmin([dist_fn(c) for c in candidates]))]

        forward = target - cam_pos
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:  # degenerate: forward ≈ ±Z
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = cam_pos

        cam_node = render_scene.add(camera, pose=pose)
        color, _ = renderer.render(render_scene, flags=pyrender.RenderFlags.FLAT)
        render_scene.remove_node(cam_node)

        out_path = os.path.join(out_dir, fname)
        Image.fromarray(color).save(out_path)
        out_paths.append(out_path)

    renderer.delete()
    return tuple(out_paths)


def load_aligned_mesh(scene_id, data_dir):
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(data_dir, f"{scene_id}_vh_clean_2.ply")
    )
    mesh.compute_vertex_normals()

    meta_file = os.path.join(data_dir, f"{scene_id}.txt")
    with open(meta_file, "r") as f:
        lines = f.readlines()

    axis_align_matrix = None
    for line in lines:
        if "axisAlignment" in line:
            axis_align_matrix = np.array(
                [float(x) for x in line.rstrip().strip("axisAlignment = ").split()]
            ).reshape(4, 4)
            break

    if axis_align_matrix is None:
        raise ValueError("axisAlignment not found in metadata file.")

    mesh.transform(axis_align_matrix)
    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", default="scene0085_00")
    parser.add_argument("--data_dir", default="examples/data")
    parser.add_argument("--output_dir", default="examples/output")
    parser.add_argument("--max_size", type=int, default=1024)
    args = parser.parse_args()

    scene_output_dir = os.path.join(args.output_dir, args.scene_id)
    os.makedirs(scene_output_dir, exist_ok=True)

    mesh = load_aligned_mesh(args.scene_id, args.data_dir)

    extent = mesh.get_axis_aligned_bounding_box().get_extent()  # [m, n, k]

    topview_path = os.path.join(scene_output_dir, "topview.png")
    render_topview_zup_no_margin(mesh, topview_path, args.max_size)

    sideviews_dir = os.path.join(scene_output_dir, "sideviews")
    render_4_sideviews(mesh, sideviews_dir, args.max_size, args.max_size)

    with Image.open(topview_path) as topview_img:
        w, h = topview_img.size

    metadata = {"m": extent[0], "n": extent[1], "k": extent[2], "w": w, "h": h}
    with open(os.path.join(scene_output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
