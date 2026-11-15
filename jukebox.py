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

    # transform blue, green, red to hue, saturation, value
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)


# slice image into 60 (4320 x 128 x 3) pieces
def slice_image(img:np.ndarray) -> np.ndarray:
    img_s = np.empty((60, 4320, 128, 3), dtype=img.dtype)
    j = 0
    for i in range(0, 7680, 128):
        img_s[j] = img[:, i:i+128]
        j += 1

    return img_s


# removes very dark pixels so that colors can be extracted
def remove_noise(slice:np.ndarray) -> np.ndarray:
    mask = slice[:, :, 2] > 20
    slice_clean = slice[mask]
    # check whether slice has only dark pixels
    if len(slice_clean) == 0:
        # convert to 2D array
        return slice.reshape((4320 * 128, 3))
    
    return slice_clean


# determine color spectrum of current slice
def color_spectrum(slice:np.ndarray) -> str:
    slice_clean = remove_noise(slice)
    hue = slice_clean[:, 0]
    color = np.median(hue)
    if color < 30:
        return "red"
    elif color < 60:
        return "yellow"
    elif color < 90:
        return "green"
    elif color < 120:
        return "cyan"
    elif color < 150:
        return "blue"
    else:
        return "magenta"


# determine brightness of current slice
def brightness(slice:np.ndarray) -> float:
    value = slice[:, :, 2]
    # return average brightness from 0.0 to 1.0
    return float(np.mean(value) / 255)


# determine pixel_density of current slice
def pixel_density():
    pass


if __name__ == "__main__":
    file_name = "Images/Aerosols_2025-08-19T20_00_00_7680x4320.png"
    img = load_image(file_name)
    img_s = slice_image(img)
    print(brightness(img_s[30]))
