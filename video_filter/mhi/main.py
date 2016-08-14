import cv2
import numpy as np
from skimage import img_as_ubyte
from tqdm import tqdm

def main(video, param, filepath_avi ):

    sigma_blur = param['sigma_blur']
    threshold = param['threshold']
    kernel_size = param['kernel_size']
    tau = param['tau']

    (cap, nbf, fps, width, height) = video.get_cap()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(filepath_avi, fourcc, fps,
                             (width, height), True)

    ret, frame = cap.read()
    empty = np.zeros(frame.shape, np.uint8)
    # To keep same time as orig
    writer.write(empty)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * 1.0
    graybl = cv2.GaussianBlur(gray, (sigma_blur, sigma_blur), 0)

    mhi = np.zeros(gray.shape)

    for i in tqdm(range(nbf)):

        ret, frame = cap.read()
        if not ret:
            break

        graybl_old = graybl
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * 1.0
        graybl = cv2.GaussianBlur(gray, (sigma_blur, sigma_blur), 0)

        diff = graybl - graybl_old

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        B = cv2.morphologyEx((np.abs(diff) > threshold) * 1.0, cv2.MORPH_OPEN, kernel)
        mhi[B == 0] = np.maximum(0, mhi[B == 0] - 1)
        mhi[B > 0] = tau

        img_mhi = mhi / tau
        img_mhi = img_as_ubyte(img_mhi)
        img_mhi = cv2.cvtColor(img_mhi, cv2.COLOR_GRAY2BGR)
        writer.write(img_mhi)

    cap.release()
    writer.release()
