# Starline CrystalGlass 3D – Rotating Face Mesh Simulation (Upgrade)
# Simulates stereo-angle holographic projection of the face mesh inside a crystal cube

import cv2
import numpy as np
import torch
from time import time
from scipy.ndimage import gaussian_filter

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.functions import rotate_vertices

class StarlineCrystalGlass3D:
    def _init_(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Load 3DDFA
        self.face_boxes = FaceBoxes()
        self.tddfa = TDDFA(gpu_mode=(self.device.type == 'cuda'))
        
        # Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.cube_size = 0.8
        self.caustic_intensity = 0.3
        self.angle_shift = 15  # degrees for left/right rotation simulation

    def render_hologram(self, frame, verts, tri):
        h, w = frame.shape[:2]
        cube_size = int(min(h, w) * self.cube_size)
        x1, y1 = (w - cube_size) // 2, (h - cube_size) // 2
        x2, y2 = x1 + cube_size, y1 + cube_size
        cube_mask = np.zeros((h, w), dtype=np.float32)
        cube_mask[y1:y2, x1:x2] = 1

        angles = [-self.angle_shift, 0, self.angle_shift]
        rendered_layers = []

        for angle in angles:
            rotated_verts = rotate_vertices(verts.copy(), angle)
            rendered = render(frame, rotated_verts, tri, show_flag=False)
            rendered_layers.append(rendered.astype(np.float32))

        blend = np.mean(rendered_layers, axis=0)
        blurred = gaussian_filter(blend, sigma=1)
        final = (frame * (1 - cube_mask[..., None]) + blurred * cube_mask[..., None]).astype(np.uint8)

        edges = cv2.Canny((cube_mask * 255).astype(np.uint8), 100, 200)
        final[edges > 0] = [255, 255, 255]
        return final

    def run(self):
        while self.cap.isOpened():
            start = time()
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            draw_frame = frame.copy()

            boxes = self.face_boxes(frame)
            if len(boxes) > 0:
                param_lst, roi_box_lst = self.tddfa(frame, boxes)
                ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
                output = self.render_hologram(draw_frame, ver_lst[0], self.tddfa.tri)
            else:
                output = frame

            fps = 1 / (time() - start)
            cv2.putText(output, f'FPS: {fps:.1f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Starline 3D CrystalGlass", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if _name_ == "_main_":
    app = StarlineCrystalGlass3D()
    app.run()
