import numpy as np
import trimesh
import pyrender
from PIL import Image


def render_topview_zup_no_margin(o3d_mesh, output_path="topview.png", max_size=512):
    """Render a top-down (Z-up) orthographic view of an Open3D mesh and save it to output_path."""
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

    bbox = mesh.bounding_box
    center = bbox.centroid
    extent = bbox.extents

    extent_x, extent_y, extent_z = extent

    # Image size: longest side = max_size, keep mesh aspect ratio
    mesh_aspect = extent_x / extent_y

    if mesh_aspect >= 1:
        width = max_size
        height = int(max_size / mesh_aspect)
    else:
        height = max_size
        width = int(max_size * mesh_aspect)

    # Camera topview (Z-up)
    cam_height = extent_z * 2.0
    cam_pos = center + np.array([0, 0, cam_height])

    forward = center - cam_pos
    forward /= np.linalg.norm(forward)

    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_pos

    # Orthographic camera fit to mesh extent
    xmag = extent_x / 2.0
    ymag = extent_y / 2.0

    # Add a small padding (convert pixel padding to world units)
    pad_x = extent_x * (2 / width)
    pad_y = extent_y * (2 / height)

    xmag += pad_x
    ymag += pad_y

    scene = pyrender.Scene(
        bg_color=[255, 255, 255, 255],
        ambient_light=[1.0, 1.0, 1.0]
    )

    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(render_mesh)

    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
    scene.add(camera, pose=pose)

    renderer = pyrender.OffscreenRenderer(width, height)

    color, _ = renderer.render(
        scene,
        flags=pyrender.RenderFlags.FLAT
    )

    Image.fromarray(color).save(output_path)
    renderer.delete()

    return output_path
