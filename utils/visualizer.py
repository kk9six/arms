import os
import numpy as np
import gc
os.environ["OPEN3D_CPU_RENDERING"] = "true"
import open3d as o3d
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tqdm
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont


class Visualizer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def view_points(points, color=(255, 0, 0), highlight=False, radius=0.02):
        """
        :param (n, 3) or (3) points: points to be visualized
        :param rgb, (b, 3) or (3,) color: color of the points
        :param bool highlight: if True, draw a sphere at each point
        :param float radius: radius of the marker

        :return trimesh.PointCloud content: content to be visualized
        """
        points = points.reshape(-1, 3)
        if not highlight:  # pure point
            point_cloud = trimesh.PointCloud(vertices=points, colors=[color] * len(points))
            return point_cloud
        # sphere with radius 0.02
        content = []
        for p in points:
            marker = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            marker.apply_translation(p)
            marker.visual.vertex_colors = [color] * len(marker.vertices)
            content.append(marker)
        return content

    @staticmethod
    def view_skeletons(joints, kinematic_tree, color=(255, 0, 0)):
        joints = joints.reshape(-1, 3)
        content = []
        for i, p in enumerate(joints):
            marker = trimesh.creation.icosphere(subdivisions=3, radius=0.03)
            marker.apply_translation(p)
            marker.visual.vertex_colors = [color] * len(marker.vertices)
            content.append(marker)
        for i, j in zip(range(len(kinematic_tree)), kinematic_tree.tolist()):
            if i == j:
                continue
            p1 = joints[i]
            p2 = joints[j]
            v = p2 - p1
            length = np.linalg.norm(v)
            if length < 1e-8:
                continue
            cylinder = trimesh.creation.cylinder(radius=0.01, height=length, sections=12)
            z = np.array([0, 0, 1])
            v_norm = v / length
            axis = np.cross(z, v_norm)
            angle = np.arccos(np.clip(np.dot(z, v_norm), -1.0, 1.0))
            if np.linalg.norm(axis) < 1e-8:
                R = np.eye(3)
            else:
                axis = axis / np.linalg.norm(axis)
                K = np.array(
                    [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
                )
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            cylinder.apply_transform(np.vstack([np.hstack([R, p1.reshape(3, 1)]), [0, 0, 0, 1]]))
            cylinder.apply_translation(v_norm * (length / 2))
            cylinder.visual.vertex_colors = [color] * len(cylinder.vertices)
            content.append(cylinder)
        return content

    @staticmethod
    def view_mesh(vertices, faces, colors=None, wireframe=False, vertex_colors=None):
        """
        :param (b, n, 3) or (n, 3) vertices: vertices to be visualized
        :param (n_tris, 3) faces: faces of the mesh
        :param (b, 3) or (3, ) colors: color of the mesh
        :param bool wireframe: if True, draw the wireframe of the mesh

        :return trimesh.Trimesh content: content to be visualized
        """
        # reshape to (b, n ,3)
        vertices = vertices.reshape(-1, vertices.shape[-2], vertices.shape[-1])
        n_batch = vertices.shape[0]
        if colors is None:
            colors = [None] * n_batch
        else:
            colors = torch.tensor(colors)
            colors = colors.reshape(-1, colors.shape[-1])

        if vertex_colors is None:
            vertex_colors = [None] * n_batch
        else:
            vertex_colors = torch.tensor(vertex_colors)
            vertex_colors = vertex_colors.reshape(-1, vertex_colors.shape[-1])

        content = []
        for i in range(n_batch):
            mesh = trimesh.Trimesh(
                vertices=vertices[i],
                faces=faces,
                process=False,
                maintain_order=True,
                face_colors=colors[i],
                vertex_colors=vertex_colors[i],
            )
            if not wireframe:
                content.append(mesh)
                continue
            # Create 3D paths from the edges
            edges = mesh.edges
            edge_paths = trimesh.load_path(mesh.vertices[edges])
            edge_paths.colors = np.repeat([[214, 214, 214, 1]], len(edge_paths.entities), axis=0)
            content.append(edge_paths)
        return content

    @staticmethod
    def view_coordinate_frame(origin=None, R=None, axis_length=0.4):
        if origin is not None and R is not None:
            scene = trimesh.Scene()
            colors = {
                0: [255, 0, 0, 255],  # X - red
                1: [0, 255, 0, 255],  # Y - green
                2: [0, 0, 255, 255],  # Z - blue
            }
            for i in range(3):
                axis_vec = R[:, i] * axis_length
                start = origin
                end = origin + axis_vec
                cylinder = trimesh.creation.cylinder(radius=0.005, segment=[start, end], sections=8)
                cylinder.visual.vertex_colors = colors[i]
                scene.add_geometry(cylinder)
            return scene
        coordinate_frame = trimesh.creation.axis(
            origin_color=(125, 125, 125, 255), axis_radius=0.01, axis_length=axis_length
        )
        return coordinate_frame

    @staticmethod
    def view_paths(paths, color=(255, 0, 0)):
        """
        :param paths:
        :param color:
        """
        try:
            re_paths = paths.reshape(-1, paths.shape[-2], paths.shape[-1])
            p = trimesh.load_path(re_paths)
        except Exception:
            p = trimesh.load_path(paths)
        # p.colors = sns.light_palette('red', n_colors=len(paths)-1)
        p.colors = [color] * len(p.entities)
        return p

    @staticmethod
    def viewer(scene_content):
        """
        viewer(scene_content).show()
        """
        # scene = trimesh.Scene(flatten(scene_content))
        scene = trimesh.Scene(scene_content)
        return scene

    @staticmethod
    def view_motion(
        vertice_sequence,
        faces,
        trans=None,
        coord_centers=None,
        coord_yaw_angles=None,
        fname="output.mp4",
        fps=60,
        size=(1920, 1080),
    ):
        assert vertice_sequence.ndim == 3 or vertice_sequence.ndim == 4, (
            "vertice_sequence must be a 3D or 4D tensor"
        )

        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy().astype(np.int32)
        elif not isinstance(faces, np.ndarray):
            faces = np.array(faces).astype(np.int32)

        # Ensure vertice_sequence is a numpy array
        if isinstance(vertice_sequence, torch.Tensor):
            vertice_sequence = vertice_sequence.detach().cpu().numpy()
        elif not isinstance(vertice_sequence, np.ndarray):
            vertice_sequence = np.array(vertice_sequence)

        if vertice_sequence.ndim == 3:
            vertice_sequence = np.expand_dims(
                vertice_sequence, 0
            )  # to (n_person, n_frames, n_vertex, 3)

        if trans is None:
            trans = np.mean(vertice_sequence, axis=2)
        else:
            if isinstance(trans, torch.Tensor):
                trans = trans.detach().cpu().numpy()
            elif not isinstance(trans, np.ndarray):
                trans = np.array(trans)

        if trans.ndim == 2:
            trans = np.expand_dims(trans, 0)  # to (n_person, n_frames, 3)

        # Create first frame mesh
        meshes = [
            o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(vertice_sequence[0]),
                triangles=o3d.utility.Vector3iVector(faces),
            )
            for vertice_sequence in vertice_sequence
        ]
        meshes = [mesh.compute_vertex_normals() for mesh in meshes]

        # Create coordinate frame and apply yaw rotation if needed
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0] if coord_centers is None else coord_centers[0]
        )
        if coord_yaw_angles is not None:
            # R_yaw = coordinate_frame.get_rotation_matrix_from_axis_angle([0, coord_yaw_angles[0], 0])
            coordinate_frame.rotate(
                coord_yaw_angles[0], center=(0, 0, 0) if coord_centers is None else coord_centers[0]
            )

        # Calculate trajectory bounds for ground plane sizing
        min_x, min_z = trans[:, :, 0].min(), trans[:, :, 2].min()
        max_x, max_z = trans[:, :, 0].max(), trans[:, :, 2].max()

        padding = max(max_x - min_x, max_z - min_z) * 0.3
        ground_min_x, ground_max_x = min_x - padding, max_x + padding
        ground_min_z, ground_max_z = min_z - padding, max_z + padding
        ground_width = ground_max_x - ground_min_x
        ground_depth = ground_max_z - ground_min_z
        ground_y = -0.01  # Slightly below the mesh

        ground_plane = o3d.geometry.TriangleMesh()
        ground_vertices = np.array(
            [
                [ground_min_x, ground_y, ground_min_z],  # Bottom-left
                [ground_max_x, ground_y, ground_min_z],  # Bottom-right
                [ground_max_x, ground_y, ground_max_z],  # Top-right
                [ground_min_x, ground_y, ground_max_z],  # Top-left
            ]
        )
        ground_faces = np.array([[0, 1, 2], [0, 2, 3]])
        ground_plane.vertices = o3d.utility.Vector3dVector(ground_vertices)
        ground_plane.triangles = o3d.utility.Vector3iVector(ground_faces)
        ground_plane.compute_vertex_normals()

        # Create grid lines
        grid_spacing = max(ground_width, ground_depth) / 20  # 20 grid divisions
        grid_lines = []
        grid_points = []
        point_idx = 0

        # Vertical grid lines (along Z-axis)
        x = ground_min_x
        while x <= ground_max_x:
            grid_points.extend(
                [[x, ground_y + 0.001, ground_min_z], [x, ground_y + 0.001, ground_max_z]]
            )
            grid_lines.append([point_idx, point_idx + 1])
            point_idx += 2
            x += grid_spacing

        # Horizontal grid lines (along X-axis)
        z = ground_min_z
        while z <= ground_max_z:
            grid_points.extend(
                [[ground_min_x, ground_y + 0.001, z], [ground_max_x, ground_y + 0.001, z]]
            )
            grid_lines.append([point_idx, point_idx + 1])
            point_idx += 2
            z += grid_spacing

        grid_lineset = o3d.geometry.LineSet()
        grid_lineset.points = o3d.utility.Vector3dVector(grid_points)
        grid_lineset.lines = o3d.utility.Vector2iVector(grid_lines)
        grid_lineset.colors = o3d.utility.Vector3dVector(
            [[0.6, 0.6, 0.6]] * len(grid_lines)
        )  # Gray grid

        # Create offscreen renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(size[0], size[1])
        # renderer.scene.view.set_sample_count(16)

        # Add ground plane with material
        ground_mat = o3d.visualization.rendering.MaterialRecord()
        ground_mat.shader = "defaultLit"
        ground_mat.base_color = [0.7, 0.7, 0.7, 1.0]  # Light gray ground
        ground_mat.base_roughness = 0.8  # Matte finish
        renderer.scene.add_geometry("ground", ground_plane, ground_mat)

        # Add grid lines
        grid_mat = o3d.visualization.rendering.MaterialRecord()
        grid_mat.shader = "unlitLine"
        grid_mat.line_width = 1.0
        grid_mat.base_color = [0.5, 0.5, 0.5, 1.0]  # Darker gray for grid lines
        renderer.scene.add_geometry("grid", grid_lineset, grid_mat)

        # Add geometries to the scene with better materials
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"  # Use lit shader for better lighting
        mat.base_color = [0.8, 0.8, 0.8, 1.0]  # Light gray color
        for i, mesh in enumerate(meshes):
            renderer.scene.add_geometry(f"mesh_{i}", mesh, mat)

        # Coordinate frame with different material
        coord_mat = o3d.visualization.rendering.MaterialRecord()
        coord_mat.shader = "defaultUnlit"
        renderer.scene.add_geometry("coordinate_frame", coordinate_frame, coord_mat)

        # Set up lighting - multiple light sources for better quality
        renderer.scene.set_lighting(renderer.scene.LightingProfile.MED_SHADOWS, [0, 0, -1])
        renderer.scene.scene.add_directional_light(
            "sun",
            np.array([1, 1, 1], dtype=np.float32),
            np.array([1, -1, -1], dtype=np.float32),
            50000,
            True,
        )
        renderer.scene.scene.add_directional_light(
            "fill",
            np.array([0.8, 0.8, 1.0], dtype=np.float32),
            np.array([-1, -1, 1], dtype=np.float32),
            30000,
            False,
        )

        # Set background color to while
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

        # Set up camera view for multiple meshes
        bounds = [mesh.get_axis_aligned_bounding_box() for mesh in meshes]
        # Compute the overall bounding box that contains all meshes
        overall_bbox = bounds[0]
        for b in bounds[1:]:
            overall_bbox += b
        center = overall_bbox.get_center()
        extent = overall_bbox.get_extent()

        # Use the overall bounding box for camera parameters
        eye, up = Visualizer.get_camera_params(
            overall_bbox, elevation=30, azimuth=45, distance_scale=1.3
        )
        renderer.scene.camera.look_at(center, eye, up)

        images = []

        # Initialize trajectory data
        trajectory_points_all = [[] for _ in range(trans.shape[0])]
        traj_mat = o3d.visualization.rendering.MaterialRecord()
        traj_mat.shader = "unlitLine"
        traj_mat.line_width = 3.0
        traj_mat.base_color = [1.0, 0.0, 0.0, 1.0]  # Red trajectory

        img = renderer.render_to_image()
        images.append(np.asarray(img))

        # Render subsequent frames
        n_frames = vertice_sequence.shape[1]
        for i in tqdm(range(1, n_frames), total=n_frames):
            # Update mesh vertices
            for subject_idx, mesh in enumerate(meshes):
                mesh.vertices = o3d.utility.Vector3dVector(vertice_sequence[subject_idx, i])
                mesh.compute_vertex_normals()
                renderer.scene.remove_geometry(f"mesh_{subject_idx}")
                renderer.scene.add_geometry(f"mesh_{subject_idx}", mesh, mat)

            if coord_centers is not None:
                # coordinate_frame.set_origin(coord_centers[i])
                # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                #     size=0.3, origin=coord_centers[i]
                # )
                coordinate_frame.translate(coord_centers[i], relative=False)
                if coord_yaw_angles is not None:
                    # R_yaw = coordinate_frame.get_rotation_matrix_from_axis_angle([0, coord_yaw_angles[i], 0])
                    # coordinate_frame.rotate(R_yaw, center=(0, 0, 0))
                    coordinate_frame.rotate(coord_yaw_angles[i], center=(coord_centers[i]))
                renderer.scene.remove_geometry("coordinate_frame")
                renderer.scene.add_geometry("coordinate_frame", coordinate_frame, coord_mat)

            # Update trajectory
            for subject_idx in range(trans.shape[0]):
                trajectory_points = trajectory_points_all[subject_idx]
                trajectory_points.append(trans[subject_idx, i])

                # Only create and add trajectory if there are at least 2 points
                if len(trajectory_points) < 2:
                    continue

                lines = []
                colors = []
                for k in range(len(trajectory_points) - 1):
                    lines.append([k, k + 1])
                    # Gradient color from light red to dark red (older to newer)
                    alpha = 1.0 - (k + 1) / len(trajectory_points)
                    colors.append(
                        [1.0 - 0.5 * alpha, 0.0, 0.0]
                    )  # from light red ([1,0.5,0]) to dark red ([0.5,0,0])

                # Only create LineSet if there are lines to draw
                if len(lines) > 0:
                    trajectory_lines = o3d.geometry.LineSet()
                    trajectory_lines.points = o3d.utility.Vector3dVector(trajectory_points)
                    trajectory_lines.lines = o3d.utility.Vector2iVector(lines)
                    trajectory_lines.colors = o3d.utility.Vector3dVector(colors)

                    # Add or update trajectory in renderer
                    if i > 0:
                        try:
                            renderer.scene.remove_geometry(f"trajectory_{subject_idx}")
                        except Exception:
                            pass  # Ignore if geometry does not exist

                    try:
                        renderer.scene.add_geometry(
                            f"trajectory_{subject_idx}", trajectory_lines, traj_mat
                        )
                    except Exception as e:
                        print(f"Failed to add trajectory_{subject_idx}: {e}")

            # Update camera to always point to the new center
            bounds = [mesh.get_axis_aligned_bounding_box() for mesh in meshes]
            overall_bbox = bounds[0]
            for b in bounds[1:]:
                overall_bbox += b
            center = overall_bbox.get_center()
            extent = overall_bbox.get_extent()

            # # Keep same camera distance but update target center
            # camera_distance = extent.max() * .8
            # eye = center + np.array([camera_distance, camera_distance, camera_distance])

            # eye, up = Visualizer.get_camera_params(
            #     overall_bbox, elevation=30, azimuth=45, distance_scale=1.3
            # )
            # renderer.scene.camera.look_at(center, eye, up)

            # Render frame
            img = renderer.render_to_image()
            images.append(np.asarray(img))

        # if save_video:
        clip = ImageSequenceClip(images, fps=fps)
        clip.write_videofile(fname)
        clip.close()
        # if return_images:
        #     return images
        # else:
        #     del images
        # del renderer
        del renderer
        # gc.collect()

    @staticmethod
    def get_camera_params(bounds, elevation=45, azimuth=45, distance_scale=1):
        center = bounds.get_center()
        extent = bounds.get_extent()

        d = np.max(bounds.get_extent()) * distance_scale  # zoom out a bit
        elev = np.deg2rad(elevation)
        azim = np.deg2rad(azimuth)

        eye = center + np.array(
            [
                d * np.cos(elev) * np.cos(azim),  # X
                d * np.sin(elev),  # Y (height)
                d * np.cos(elev) * np.sin(azim),  # Z
            ]
        )
        up = [0, 1, 0]

        return eye, up

    @staticmethod
    def export_skeleton(
        joints_sequence,
        kinetree,
        fname="output.mp4",
        fps=30,
        size=(1920, 1080),
        joint_radius=0.04,
        bone_radius=0.02,
        joint_color=[[0.2, 0.6, 1.0], [1.0, 0.6, 0.2]],
        bone_color=[[0.8, 0.2, 0.2], [0.2, 0.2, 0.8]],
        timeline=None,
        verbose=True,
        text=None,  # Add text parameter
        text_position=(50, 50),  # Add position parameter
        text_color=(0, 0, 0),  # Add color parameter
        text_size=72,  # Add font size parameter
    ):
        kinetree = list(zip(range(len(kinetree)), kinetree.tolist()))

        # Convert to numpy
        if isinstance(joints_sequence, torch.Tensor):
            joints_sequence = joints_sequence.detach().cpu().numpy()
        elif not isinstance(joints_sequence, np.ndarray):
            joints_sequence = np.array(joints_sequence)

        if joints_sequence.ndim == 3:
            joints_sequence = np.expand_dims(
                joints_sequence, 0
            )  # (n_person, n_frames, n_joints, 3)

        if timeline is None:
            timeline = np.ones((joints_sequence.shape[:2]), dtype=bool)  # (n_person, n_frames)

        assert timeline.shape == joints_sequence.shape[:2], "timeline must have the same shape as joints_sequence"

        n_person, n_frames, n_joints, _ = joints_sequence.shape
        writer = FFMPEG_VideoWriter(
            fname,
            (1920, 1080),
            fps=fps,
            preset="veryslow",
            codec="libx264",
            # bitrate="50M",
            ffmpeg_params=["-crf", "15"],
        )

        mat_joint = o3d.visualization.rendering.MaterialRecord()
        mat_joint.shader = "defaultLit"
        mat_joint.base_roughness = 0.1  # Slight glossiness
        mat_joint.base_metallic = 0.1

        mat_bone = o3d.visualization.rendering.MaterialRecord()
        mat_bone.shader = "defaultLit"
        mat_bone.base_roughness = 0.5
        mat_bone.base_metallic = 0.0

        mat_ground = o3d.visualization.rendering.MaterialRecord()
        mat_ground.shader = "defaultLit"
        mat_ground.base_roughness = 0.8
        mat_ground.base_metallic = 0.0
        mat_ground.base_color = [0.4, 0.4, 0.4, 1.0]

        # Add a ground plane (a large gray rectangle at y=0)
        # Set ground_size based on the motion's spatial extent
        all_joints = joints_sequence  # (n_person, n_frames, n_joints, 3)
        min_xyz = all_joints.reshape(-1, 3).min(axis=0)
        max_xyz = all_joints.reshape(-1, 3).max(axis=0)
        ground_center = (min_xyz + max_xyz) / 2
        ground_width = max((max_xyz[0] - min_xyz[0]) + 1, 4)
        ground_depth = max((max_xyz[2] - min_xyz[2]) + 1, 2)
        ground_height = 0.01  # y position of the ground plane
        ground = o3d.geometry.TriangleMesh.create_box(
            width=ground_width, height=ground_height, depth=ground_depth
        )
        # Move the ground so its center aligns with ground_center
        ground.translate(
            [
                ground_center[0] - ground_width / 2,
                -ground_height / 2,
                ground_center[2] - ground_depth / 2,
            ]
        )
        ground.paint_uniform_color([0.3, 0.3, 0.3])  # light gray
        ground.compute_vertex_normals()

        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        coordinate_frame.compute_vertex_normals()

        # Set camera parameters so the ground plane is in the middle of the view
        center = ground_center.copy()
        center[1] = 0.0  # keep camera looking at y=0 for ground
        extent = max(ground_width, ground_depth)
        cam_dist = extent * .9 # 1.1  # heuristic: step back to capture everything
        angle = 45 # 45  # field of view degrees
        # Camera eye: above and off to the side so as to see ground and motion
        # eye = center + np.array([0.8 * cam_dist, cam_dist, 0.8 * cam_dist])
        eye = center + np.array([.8 * cam_dist, 1.0 * cam_dist, .8 * cam_dist])
        up = [0, 1, 0]  # y up

        render = o3d.visualization.rendering.OffscreenRenderer(size[0], size[1])
        render.scene.set_lighting(
            o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
            (0.577, -0.577, -0.577),
        )
        render.scene.add_geometry("ground", ground, mat_ground)
        render.scene.add_geometry("coord", coordinate_frame, mat_ground)
        render.setup_camera(angle, center, eye, up)

        # try:
            # font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", text_size)
        font = ImageFont.load_default(size=20)
        # except:
        #     try:
        #         font = ImageFont.truetype("arial.ttf", text_size)
        #     except:
        #         font = ImageFont.load_default()

        for f in tqdm(range(n_frames), total=n_frames, disable=not verbose):
            for p in range(n_person):
                if not timeline[p, f]:
                    continue
                joints = joints_sequence[p, f]
                for j in range(n_joints):
                    name = f"joint_{p}_{j}"
                    render.scene.remove_geometry(name)
                    sphere = o3d.geometry.TriangleMesh.create_sphere(
                        radius=joint_radius, resolution=30
                    )
                    sphere.paint_uniform_color(joint_color[p])
                    sphere.translate(joints[j])
                    sphere.compute_vertex_normals()
                    render.scene.add_geometry(name, sphere, mat_joint)
                for b_idx, (j0, j1) in enumerate(kinetree):
                    if j0 == j1:
                        continue
                    pt0 = joints[j0]
                    pt1 = joints[j1]
                    name = f"bone_{p}_{b_idx}"
                    render.scene.remove_geometry(name)
                    height = np.linalg.norm(pt1 - pt0)
                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                        radius=bone_radius, height=height, resolution=30
                    )
                    cylinder.paint_uniform_color(bone_color[p])
                    direction = pt1 - pt0
                    if np.linalg.norm(direction) > 1e-6:
                        direction = direction / np.linalg.norm(direction)
                        z_axis = np.array([0, 0, 1])
                        v = np.cross(z_axis, direction)
                        c = np.dot(z_axis, direction)
                        if np.linalg.norm(v) < 1e-6:
                            R = np.eye(3)
                        else:
                            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                            R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
                        cylinder.rotate(R, center=np.zeros(3))
                    cylinder.translate((pt1 + pt0) / 2)
                    cylinder.compute_vertex_normals()
                    render.scene.add_geometry(name, cylinder, mat_bone)
            img = np.asarray(render.render_to_image()).astype(np.uint8)
            # Add text if provided
            if text is not None:
                # Convert to PIL Image for text drawing
                if isinstance(text, list):
                    t = text[f]
                else:
                    t = text

                pil_img = Image.fromarray(img)
                draw = ImageDraw.Draw(pil_img)
                max_text_width = 850  # pixels
                x, y = 300, 64
                line_spacing = 18

                lines = wrap_text(draw, t, font, max_text_width)

                for i, line in enumerate(lines):
                    yy = y + i * (font.size + line_spacing)
                    draw.text((x, yy), line, font=font, fill=(0, 0, 0))

                img = np.array(pil_img)
            writer.write_frame(img)
        writer.close()
        del render
        gc.collect()

def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = current + (" " if current else "") + word
        w = draw.textbbox((0, 0), test, font=font)[2]
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines
