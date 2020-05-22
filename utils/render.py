import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import numpy as np
import trimesh
import pyrender
import glob


def render_trimesh(trimesh_mesh, eye, center, world_up, res=(640, 640), light_intensity=3.0, **kwargs):
    """Render a shapenet mesh using default settings.
    
    Args:
      trimesh_mesh: trimesh mesh instance, or a list of trimesh meshes (or point clouds).
      eye: array with shape [3,] containing the XYZ world
          space position of the camera.
      center: array with shape [3,] containing a position
          along the center of the camera's gaze.
      world_up: np.float32 array with shape [3,] specifying the
          world's up direction; the output camera will have no tilt with respect to
          this direction.
      res: 2-tuple of int, [width, height], resolution (in pixels) of output images.
      light_intensity: float, light intensity.
      kwargs: additional flags to pass to pyrender renderer.
    Returns:
      color_img: [*res, 3] color image.
      depth_img: [*res, 1] depth image.
      world_to_cam: [4, 4] camera to world matrix.
      projection_matrix: [4, 4] projection matrix, aka cam_to_img matrix.
    """
    if not isinstance(trimesh_mesh, list):
        trimesh_mesh = [trimesh_mesh]
    eye = list2npy(eye).astype(np.float32)
    center = list2npy(center).astype(np.float32)
    world_up = list2npy(world_up).astype(np.float32)
    
    # setup camera pose matrix
    scene = pyrender.Scene()
    for tmesh in trimesh_mesh:
        if not (isinstance(tmesh, trimesh.Trimesh) or isinstance(tmesh, trimesh.PointCloud)):
            raise NotImplementedError("All instances in trimesh_mesh must be either trimesh.Trimesh or "
                                      f"trimesh.PointCloud. Instead it is {type(tmesh)}.")
        if isinstance(tmesh, trimesh.Trimesh):
            mesh = pyrender.Mesh.from_trimesh(tmesh)
        else:
            if tmesh.colors is not None:
                colors = np.array(tmesh.colors)
            else:
                colors = np.ones_like(tmesh.vertices)
            mesh = pyrender.Mesh.from_points(np.array(tmesh.vertices), colors=colors)
        scene.add(mesh)
    
    # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    
    world_to_cam = look_at(eye[None], center[None], world_up[None])
    world_to_cam = world_to_cam[0]
    cam_pose = np.linalg.inv(world_to_cam)
    scene.add(camera, pose=cam_pose)

    # Set up the light -- a single spot light in the same spot as the camera
    light = pyrender.SpotLight(color=np.ones(3, dtype=np.float32), intensity=light_intensity,
                               innerConeAngle=np.pi/16.0)
    scene.add(light, pose=cam_pose)
    
    # Render the scene
    r = pyrender.OffscreenRenderer(*res, **kwargs)
    color_img, depth_img = r.render(scene)
    return color_img, depth_img, world_to_cam, camera.get_projection_matrix(*res)


def _unproject_points(points, projection_matrix, world_to_cam):
    """Unproject points from image space to world space."""
    # pad
    depth = points[:, 2]
    xy_scale = (depth - projection_matrix[2, 3]) / (-projection_matrix[2, 2])
    points[:, :2] = points[:, :2] * xy_scale[:, None]
    points = np.concatenate([points, np.ones_like(points[:, :1])], axis=1)
    points[:, 3] = xy_scale
    # camera space coordinates
    point_cam = (np.linalg.inv(projection_matrix)@(points.T)).T  # [npoints, 4]
    # world space coordinates
    cam_to_world = np.linalg.inv(world_to_cam)
    point_world = (cam_to_world@(point_cam.T)).T
    point_world = point_world[:, :3] / point_world[:, 3:]
    return point_world


def _points_from_depth(depth_img, zoffset=0):
    """Get image space points from depth image."""
    point_mask = (depth_img != 0.)
    point_mask_flat = point_mask.reshape(-1)
    w, h = depth_img.shape
    x, y = np.meshgrid(np.linspace(-1., 1., w),
                       np.linspace(-1., 1., h), indexing='ij')
    xy_img = np.stack([y, -x],
                      axis=-1)  # [w, h, 2]
    xy_flat = xy_img.reshape(-1, 2)  # [w*h, 2]
    point_img = xy_flat[point_mask_flat]  # [npoints, 2]
    depth = depth_img.reshape(-1)[point_mask_flat]+zoffset
    point_img = np.concatenate([point_img, depth[..., None]], axis=-1)
    return point_img


def unproject_depth_img(depth_img, projection_matrix, world_to_cam):
    """Unproject depth image to point cloud in world coordinates.
    
    Args:
      depth_img: array of [width, height] depth image.
      projection_matrix: array of [4, 4], projection matrix, aka cam_to_img matrix.
      world_to_cam: array of [4, 4], world to cam matrix, inverse of camera pose.
      
    Returns:
      point_world: array of [npoints, 3] depth scan point cloud in world coordinates.
    """
    point_img = _points_from_depth(depth_img, zoffset=projection_matrix[2, 3])
    point_world = _unproject_points(point_img, projection_matrix, world_to_cam)
    
    return point_world


def list2npy(array):
    return array if isinstance(array, np.ndarray) else np.array(array)

def r4pad(array):
    """pad [..., 3] array to [..., 4] with ones in last channel."""
    zeros = np.ones_like(array[..., -1:])
    return np.concatenate([array, zeros], axis=-1)

def look_at(eye, center, world_up):
    """Computes camera viewing matrices (numpy implementation).

    Functionality mimes gluLookAt (third_party/GL/glu/include/GLU/glu.h).
    Reference tf implementation:
    google3/research/vision/viscam/diffren/common/camera_utils.py

    Args:
    eye: np.float32 array with shape [batch_size, 3] containing the XYZ world
      space position of the camera.
    center: np.float32 array with shape [batch_size, 3] containing a position
      along the center of the camera's gaze.
    world_up: np.float32 array with shape [batch_size, 3] specifying the
      world's up direction; the output camera will have no tilt with respect to
      this direction.

    Returns:
    A [batch_size, 4, 4] np.float32 array containing a right-handed camera
    extrinsics matrix that maps points from world space to points in eye space.
    """
    batch_size = center.shape[0]
    vector_degeneracy_cutoff = 1e-6
    forward = center - eye
    forward_norm = np.linalg.norm(forward, axis=1, keepdims=True)
    assert(np.all(forward_norm > vector_degeneracy_cutoff))
    forward /= forward_norm

    to_side = np.cross(forward, world_up)
    to_side_norm = np.linalg.norm(to_side, axis=1, keepdims=True)
    assert(np.all(to_side_norm > vector_degeneracy_cutoff))
    to_side /= to_side_norm
    cam_up = np.cross(to_side, forward)

    w_column = np.array(batch_size * [[0., 0., 0., 1.]],
                      dtype=np.float32)  # [batch_size, 4]
    w_column = w_column.reshape([batch_size, 4, 1])
    view_rotation = np.stack(
      [to_side, cam_up, -forward,
       np.zeros_like(to_side, dtype=np.float32)],
      axis=1)  # [batch_size, 4, 3] matrix
    view_rotation = np.concatenate([view_rotation, w_column],
                                   axis=2)  # [batch_size, 4, 4]

    identity_batch = np.tile(np.expand_dims(np.eye(3), 0), [batch_size, 1, 1])
    view_translation = np.concatenate([identity_batch,
                                       np.expand_dims(-eye, 2)], 2)
    view_translation = np.concatenate(
        [view_translation,
         w_column.reshape([batch_size, 1, 4])], 1)
    camera_matrices = np.matmul(view_rotation, view_translation)
    return camera_matrices
