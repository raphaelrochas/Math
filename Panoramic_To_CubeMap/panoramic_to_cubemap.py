import cv2 
import numpy as np 
import math

def pano_to_cubemap(img, size):
    
    # Image source
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    assert w == 2 * h
    faces = {
        "right": np.array([1,0,0]),
        "left": np.array([-1,0,0]),
        "top": np.array([0,1,0]),
        "bottom": np.array([0,-1,0]),
        "front": np.array([0,0,1]),
        "back": np.array([0,0,-1]),
    }
    
    results = {}
    
    for name, center_dir in faces.items():
        # Grille size**2
        u = np.linspace(-1, 1, size)
        v = np.linspace(-1, 1, size)
        uu, vv = np.meshgrid(u, -v)  # inversion de l'axe y pour openCV 

        # Direction 3d
        if name == "right":
            dirs = np.stack([np.ones_like(uu), vv, uu], axis=-1)
        elif name == "left":
            dirs = np.stack([-np.ones_like(uu), vv, -uu], axis=-1)
        elif name == "top":
            dirs = np.stack([uu, np.ones_like(uu), vv], axis=-1)
        elif name == "bottom":
            dirs = np.stack([uu, -np.ones_like(uu), -vv], axis=-1)
        elif name == "front":
            dirs = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
        elif name == "back":
            dirs = np.stack([-uu, vv, -np.ones_like(uu)], axis=-1)

        # Projection 
        norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs /= norm

        lon = np.arctan2(dirs[..., 0], dirs[..., 2])
        lat = np.arcsin(dirs[..., 1])

        # Mapping
        x = (lon / math.pi + 1) * 0.5 * w
        y = (0.5 - lat / (math.pi / 2)) * h

        # Interpolation 
        face_img = cv2.remap(img, x.astype(np.float32), y.astype(np.float32),
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_WRAP)

        results[name] = face_img
        cv2.imwrite(f"{name}.jpg", face_img)

    return results


pano_to_cubemap("Panoramic_To_CubeMap/cubemap_example/kloofendal_48d_partly_cloudy_puresky_4k.jpg", 1024)
    