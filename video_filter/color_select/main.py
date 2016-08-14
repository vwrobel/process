import cv2
import numpy as np
from tqdm import tqdm


def main(video, param, filepath_avi ):


    lower = np.array(param['lower'], dtype='uint8') #lower color boundary
    upper = np.array(param['upper'], dtype='uint8') #upper color boundary
    background_color = (255, 255, 255)


    (cap, nbf, fps, width, height) = video.get_cap()

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(filepath_avi, fourcc, fps,
                             (width, height), True)

    ret, frame = cap.retrieve()
    background = np.zeros(frame.shape, np.uint8)
    background[:, :] = background_color

    for i in tqdm(range(nbf)):

        ret, frame = cap.read()
        if not ret:
            break

        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask=mask)
        output = output + cv2.add(background, output, mask =cv2.bitwise_not(mask))

        writer.write(output)

    cap.release()
    writer.release()