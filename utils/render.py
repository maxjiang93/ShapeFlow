import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np
import trimesh
import pyrender
import glob


def render_mesh(meshname, dist=1.1, res=(640, 640), color=[0., 1., 0.], intensity=3.0):
    """Render a shapenet mesh using default settings.
    
    Args:
      meshname: str, name of the mesh file.
      dist: float, camera distance from the object.
      res: 2-tuple of int, resolution of output images.
      color: 3-tuple of float in [0, 1]. color of rendered object.
    Returns:
      color_img: [*res, 3] color image.
      depth_img: [*res, 1] depth image.
    """
    trimesh_mesh = trimesh.load(meshname)
    # rotate mesh
    trimesh_mesh.apply_transform(trimesh.transformations.rotation_matrix(2.6*np.pi/2, [0, 1, 0]))
    trimesh_mesh.apply_transform(trimesh.transformations.rotation_matrix(0.3*np.pi/2, [1, 0, 0]))

    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)

    # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
           [1, 0, 0,  0.0],
           [0, 1, 0,  0.0],
           [0, 0, 1, dist],
           [0, 0, 0,  1.0],
        ])
    camera_pose = np.array(camera_pose)
    scene.add(camera, pose=camera_pose)

    # Set up the light -- a single spot light in the same spot as the camera
    light = pyrender.SpotLight(color=np.array(color), intensity=intensity,
                               innerConeAngle=np.pi/16.0)
    scene.add(light, pose=camera_pose)

    # Render the scene
    r = pyrender.OffscreenRenderer(*res)
    color_img, depth_img = r.render(scene)
    return color_img, depth_img
