import cv2
import numpy as np
import os
import csv
image_folder = r"C:\\Users\\VOVCHEK\Desktop\\ml\sphere_sfm"
output_ply = "output.ply"
output_csv = "camera_trajectory.csv"
images = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.jpg')])
if not images:
    raise Exception("Нет изображений в папке")
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
camera_poses = []  # Для хранения позиций камеры
point_cloud = []   # Для хранения 3D точек
focal_length = 6.7  # Эквивалентное фокусное расстояние в мм
sensor_width = 36.0  # Ширина сенсора в мм (приблизительно для камеры с полнокадровым сенсором)
image_width = 11968  # Разрешение по ширине

# Вычисление фокусного расстояния в пикселях
focal_px = (focal_length / sensor_width) * image_width

# Матрица калибровки камеры
K = np.array([[focal_px, 0, image_width / 2],
              [0, focal_px, image_width / 2],
              [0, 0, 1]])

dist_coeffs = None  # Коэффициенты дисторсии
for i in range(len(images) - 1):
    img1 = cv2.imread(images[i], 0)
    img2 = cv2.imread(images[i + 1], 0)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

    if len(camera_poses) == 0:
        camera_poses.append((np.eye(3), np.zeros((3, 1))))

    prev_R, prev_t = camera_poses[-1]
    curr_R = R @ prev_R
    curr_t = prev_t + prev_R @ t
    camera_poses.append((curr_R, curr_t))

    P1 = K @ np.hstack((prev_R, prev_t))
    P2 = K @ np.hstack((curr_R, curr_t))
    pts4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts4D /= pts4D[3]  # Нормализация
    point_cloud.append(pts4D[:3].T)
# Сохранение облака точек в формате .ply
with open(output_ply, 'w') as f:
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {sum(len(pc) for pc in point_cloud)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("end_header\n")
    for pc in point_cloud:
        for p in pc:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

# Сохранение траектории камеры в формате .csv
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Index', 'X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ'])
    for i, (R, t) in enumerate(camera_poses):
        rvec, _ = cv2.Rodrigues(R)
        writer.writerow([i, t[0][0], t[1][0], t[2][0], rvec[0][0], rvec[1][0], rvec[2][0]])

