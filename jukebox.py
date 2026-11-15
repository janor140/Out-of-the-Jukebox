import numpy as np
import cv2 as cv


# loads 8K image and stores it as a numpy ndarray
def load_image(file_name:str) -> np.ndarray:
    # make sure image was loaded correctly
    img = cv.imread(file_name)
    if img is None:
        raise FileNotFoundError("Path does not exist!")

    # check dimensions
    dim = img.shape
    if dim != (4320, 7680, 3):
        raise ValueError("Incorrect resolution. 8K (7680 x 4320) image expected!")

    return img


# slice image into 60 (4320 x 128 x 3) pieces
def slice_image(img:np.ndarray) -> np.ndarray:
    img_s = np.empty((60, 4320, 128, 3), dtype=img.dtype)
    j = 0
    for i in range(0, 7680, 128):
        img_s[j] = img[:, i:i+128]
        j += 1

    return img_s


if __name__ == "__main__":
    file_name = "Images/Aerosols_2025-08-19T20_00_00_7680x4320.png"
    img = load_image(file_name)
    img_s = slice_image(img)
