import sys, getopt
sys.path.append("/opt/nmr/build")
import pynmr as nmr
import cv2
import numpy as np
import quaternion
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

W = 1280
H = 720

HELP = """
Usage: python3 render.py -n <msgpack file> -m <gltf file> -l <left temple vertex> -r <right temple vertex>
Render a NeRF of a human face together with mesh glasses.

  -n, --nerf          Snapshot of previously trained NeRF model (.msgpack)
  -m, --mesh          Mesh in GLTF format. Glasses base (vertex sitting on nose) has to be at (0, 0, 0).
  -l, --left_temple   Local mesh vertex of left glasses temple. Format: "x y z"
  -r, --right_temple  Local mesh vertex of right glasses temple. Format: "x y z"

Example:
  python3 render.py -n nerf.msgpack -m glasses.gltf -l "-0.732 -1.002 -0.057" -r "0.732 -1.002 -0.057"
"""

# glasses
# -l "-0.732 -1.002 -0.057" -r "0.732 -1.002 -0.057"
# glasses 2
# -l "-351.7 -474.9 45" -r "351.7 -474.9 45"

def landmarks_to_array(landmarks):
    landmarks = landmarks.multi_face_landmarks[0].landmark
    return np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks], dtype=np.float32)

def align_point_sets(P, K):
    centroid_p = np.mean(P, axis=0)
    centroid_k = np.mean(K, axis=0)
    centered_p = P - centroid_p
    centered_k = K - centroid_k
    U, _, Vt = np.linalg.svd(centered_p.T @ centered_k)
    rotation_mat = U @ Vt
    translation = centroid_k - rotation_mat @ centroid_p
    transform = np.eye(4)
    transform[:3, :3] = rotation_mat
    transform[:3, 3] = translation
    return transform

def kabsch(P, K):
    cov_mat = np.zeros((3, 3))
    for i in range(len(P)):
        cov_mat += np.outer(P[i], K[i])
    U,_,Vt = np.linalg.svd(cov_mat)
    rotation_mat = np.dot(Vt.T, U.T)
    if np.linalg.det(rotation_mat) < 0:
        ref = np.eye(3)
        ref[-1, -1] = -1
        rotation_mat = np.dot(Vt.T, np.dot(ref, U.T))
    return quaternion.from_rotation_matrix(rotation_mat)

def render_image(nerf):
    im = np.uint8(nerf.render(W, H, linear=False) * 255)
    im = cv2.cvtColor(im[::-1, :], cv2.COLOR_BGR2RGB)
    return im

def rotate_camera_to_face_face(renderer, nerf):
    reference_landmarks = np.load("reference_landmarks.npy")
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        for i in range(5):
            i = 0
            while renderer.frame():
                im = render_image(nerf)
                results = face_mesh.process(im)
                # find first camera view in which mediapipe detects a face
                if not results.multi_face_landmarks:
                    # bruteforce camera view
                    i += 1
                    renderer.orbit(0.1, 0, np.sin(i))
                    continue
                # compute transform to match reference landmarks
                transform = align_point_sets(reference_landmarks,
                                landmarks_to_array(results))
                # orbit around the head to have a straight view on the face
                azimuth = np.arctan2(transform[0, 2], transform[0, 0])
                polar = np.arctan2(transform[2, 2], transform[1, 2]) - np.pi / 2
                renderer.orbit(-azimuth, polar, 0)
                break


class Ray:
    def __init__(self, cam_transform, landmark):
        self.origin = np.squeeze(np.array(np.transpose(cam_transform[:, 3])))
        self.dir = np.squeeze(np.array(cam_transform[0:3, 0:3].dot(
            np.array([2 * landmark.x - 1, -2 * landmark.y + 1, 1]))))

    def closest(self, ray):
        A = self.origin
        a = self.dir
        B = ray.origin
        b = ray.dir
        c = B - A
        return A + a * (- a.dot(b) * b.dot(c) + a.dot(c) * b.dot(b)) / (a.dot(a) * b.dot(b) - a.dot(b) * a.dot(b))


def closest_point_between_rays(rays):
    all_ray_pairs = [(a, b) for idx, a in enumerate(rays) for b in rays[idx + 1:]]
    the_point = np.array([0, 0, 0], dtype=np.float64)
    for ray_pair in all_ray_pairs:
        D = ray_pair[0].closest(ray_pair[1])
        E = ray_pair[1].closest(ray_pair[0])
        the_point += D + E
    return the_point / (len(all_ray_pairs) * 2)


def find_3d_landmarks(renderer, nerf):
    rotate_camera_to_face_face(renderer, nerf)
    
    rays_per_landmark = [[], [], [], [], [], [], [], [], []]

    azimuth_start = np.deg2rad(60)
    polar_start = np.deg2rad(-15)
    step = 0.05
    renderer.orbit(azimuth_start, polar_start, 0)
    renderer.orbit(0, 0, 2)
    renderer.orbit(-np.pi / 2, 0, 0)
    
    renderer.frame()
    

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
            
        j = 0
        for i in np.arange(0, np.pi, step):
            polar_step = step * np.deg2rad(40/2)
            azimuth_step = step * np.deg2rad(60/2)
            j += 1
            renderer.orbit(np.sin(i*3)*azimuth_step*3, np.sin(i)*polar_step, 0)
            renderer.frame()
            
            im = render_image(nerf)
            annotated_im = im.copy()
            results = face_mesh.process(im)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                # mp_drawing.draw_landmarks(
                #     image=annotated_im,
                #     landmark_list=results.multi_face_landmarks[0],
                #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_tesselation_style())
                # mp_drawing.draw_landmarks(
                #     image=annotated_im,
                #     landmark_list=results.multi_face_landmarks[0],
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles
                #     .get_default_face_mesh_contours_style())
                # cv2.imwrite("asd" + str(j) + ".png", annotated_im)
                transform = renderer.view_projection_mat
                rays_per_landmark[0].append(Ray(transform, landmarks[6]))   # nose
                rays_per_landmark[1].append(Ray(transform, landmarks[197])) # nose
                rays_per_landmark[2].append(Ray(transform, landmarks[195])) # nose
                rays_per_landmark[3].append(Ray(transform, landmarks[162])) # temple left
                rays_per_landmark[4].append(Ray(transform, landmarks[389])) # temple right
                rays_per_landmark[5].append(Ray(transform, landmarks[127])) # temple lower left
                rays_per_landmark[6].append(Ray(transform, landmarks[356])) # temple lower right
                rays_per_landmark[7].append(Ray(transform, landmarks[33])) # eye left
                rays_per_landmark[8].append(Ray(transform, landmarks[263])) # eye right
            
            #renderer.orbit(step, 0, 0)
            
        print(len(rays_per_landmark[0]))

    return [closest_point_between_rays(rays) for rays in rays_per_landmark]

def line_plane_intersection(line_p1, line_p2, plane_p, plane_n):
    line_d = line_p2 - line_p1
    t = np.dot(plane_n, plane_p - line_p1) / np.dot(plane_n, line_d)
    p = line_p1 + t * line_d
    return p

def place_glasses(renderer, file_path, landmarks, glasses_left, glasses_right):
    
    eye_l = landmarks[7]
    eye_r = landmarks[8]
    eye_vec = eye_l - eye_r
    eye_dist = np.linalg.norm(eye_vec)
    eye_vec = eye_vec / eye_dist
    forward_vec = np.cross(eye_vec, np.array([0, 1, 0]))
    normal_vec = np.cross(eye_vec, forward_vec)
    normal_vec = normal_vec / np.linalg.norm(normal_vec)

    left_proj = line_plane_intersection(landmarks[5], landmarks[3], eye_l, normal_vec) + forward_vec * eye_dist * 0.5
    right_proj = line_plane_intersection(landmarks[6], landmarks[4], eye_l, normal_vec) + forward_vec * eye_dist * 0.5
        
    temple_dist = np.linalg.norm(landmarks[3] - landmarks[4])
    glasses_dist = np.linalg.norm(glasses_left - glasses_right)
    scale = temple_dist / glasses_dist    
    
    rot = kabsch(
        [glasses_left, glasses_right],
        [(left_proj - landmarks[0]) / scale, (right_proj - landmarks[0]) / scale]
    )
    
    print("t=", landmarks[0], "s=", np.array([scale, scale, scale]), "r=", np.array([rot.w, rot.x, rot.y, rot.z]))
    
    return renderer.load_mesh(
        file_path,
        t = landmarks[0],
        s = np.array([scale, scale, scale]),
        r = np.array([rot.w, rot.x, rot.y, rot.z])
    )

def render(nerf_file, mesh_file, glasses_left, glasses_right):
    renderer = nmr.NerfMeshRenderer(W, H)
    renderer.envmap("sunflowers_puresky_1k.png")

    nerf = renderer.load_nerf(nerf_file)

    #renderer.remove_floaties()

    nerf.render_aabb.min = np.array([-0.2, 0.15, -0.2])
    nerf.render_aabb.max = np.array([1, 1, 1])

    print("Finding 3d face landmarks...")
    landmarks = find_3d_landmarks(renderer, nerf)
    for i,landmark in enumerate(landmarks):
       if i == 1 or i == 2: 
           continue
       #mesh = renderer.load_mesh(
       #    "./sphere.gltf", 
       #    t = landmark,
       #    s = np.array([0.015, 0.015, 0.015]))
        
    mesh = place_glasses(renderer, mesh_file, landmarks, glasses_left, glasses_right)

    a = 0
    t = time.time()
    frame_counter = 0
    while renderer.frame():
        a+= 0.03
        renderer.orbit(-(np.sin(a*1.733)) / 100 , np.cos(a*1.733) / 200, 0)
        frame_counter += 1
        new_t = time.time()
        if new_t - t >= 10:
            print("avg frame time [ms]:", (new_t - t) / frame_counter * 1000)
            t = new_t
            frame_counter = 0
        pass


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hn:m:l:r:", ["nerf=", "mesh=", "left_temple=", "right_temple="])

    nerf_file = None
    mesh_file = None
    glasses_left = None
    glasses_right = None

    for opt, arg in opts:
        if opt == "-h":
            print(HELP)
            sys.exit()
        elif opt in ("-n", "--nerf"):
            nerf_file = arg
        elif opt in ("-m", "--mesh"):
            mesh_file = arg
        elif opt in ("-l", "--left_temple"):
            glasses_left = np.fromstring(arg, dtype=float, sep=" ")
        elif opt in ("-r", "--right_temple"):
            glasses_right = np.fromstring(arg, dtype=float, sep=" ")
    
    if nerf_file is None or mesh_file is None or glasses_left is None or glasses_right is None:
        print(HELP)
        sys.exit()

    render(nerf_file, mesh_file, glasses_left, glasses_right)


