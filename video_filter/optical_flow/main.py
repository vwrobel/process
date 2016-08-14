import cv2
import numpy as np
from tqdm import tqdm

def main(video, param, filepath_avi ):

    iterations = param['iterations']
    levels = param['levels']
    pyr_scale = param['pyr_scale']
    win_size = param['win_size']
    poly_n = param['poly_n']
    poly_sigma = param['poly_sigma']

    (cap, nbf, fps, width, height) = video.get_cap()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(filepath_avi, fourcc, fps,
                             (width, height), True)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale, levels, win_size,
                                        iterations, poly_n, poly_sigma, 0)
    prvs = next

    empty = np.zeros(frame1.shape, np.uint8)
    # To keep same time as orig
    writer.write(empty)
    writer.write(empty)

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255


    for i in tqdm(range(nbf)):

        ret, frame2 = cap.read()
        if not ret:
            break

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        cv2.calcOpticalFlowFarneback(prvs, next, flow, pyr_scale, levels, win_size,
                                    iterations, poly_n, poly_sigma, cv2.OPTFLOW_USE_INITIAL_FLOW)
        prvs = next
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        img_of = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        writer.write(img_of)

    cap.release()
    writer.release()