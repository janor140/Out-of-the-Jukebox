import numpy as np
import cv2 as cv
from scipy.io import wavfile


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
def average_brightness(slice:np.ndarray) -> float:
    value = slice[:, :, 2]
    # return average brightness from 0.0 to 1.0
    return float(np.mean(value) / 255)


# determine pixel_density of current slice
def pixel_density(slice:np.ndarray) -> float:
    value = slice[:, :, 2]
    # detect edges
    edges = cv.Canny(value, 100, 200)
    return float(np.mean(edges) / 255)


# look up matching chord
def color_2_chord(color:str) -> str:
    c2c = {
        "red" : "G", "yellow": "C", "green" : "Em",
        "cyan" : "Dm", "blue" : "Am", "magenta" : "F"
    }
    return c2c.get(color)


# takes (color, brightness, density) of slice and maps it to (chord, volume, rhythm)
def sonification(color:str, brightness:float, density:float) -> list:
    chord = color_2_chord(color)
    volume = brightness
    if density > 0.5:
        rhythm = "up"
    else:
        rhythm = "down"

    return [chord, volume, rhythm]


# iterate through all slices and create audio list
def analyser(img_s:np.ndarray) -> list:
    cbd = []
    audio = []
    # collect color, brightness, and density for each slice
    for i in img_s:
        color = color_spectrum(i)
        brightness = average_brightness(i)
        density = pixel_density(i)
        cbd.append([color, brightness, density])

    # compute max values and check for edge cases
    max_brightness = max(b[1] for b in cbd)
    max_density = max(d[2] for d in cbd)
    if np.isclose(max_brightness, 0) or np.isclose(max_density, 0):
        raise ValueError("Image is too dark!")
    
    # translate to chords, volume and rhythm
    for j in cbd:
        audio.append(sonification(j[0], j[1]/max_brightness, j[2]/max_density))
    
    return audio


# converts audio list into 30 second WAV file
def create_wav(audio:list):
    # create dict that loads WAV files and stores them as ndarray
    c_dir = {
        ("Am", "down") : wavfile.read("Audio/Am[d].wav")[1], ("Am", "up") : wavfile.read("Audio/Am[u].wav")[1],
        ("C", "down") : wavfile.read("Audio/C[d].wav")[1], ("C", "up") : wavfile.read("Audio/C[u].wav")[1],
        ("Dm", "down") : wavfile.read("Audio/Dm[d].wav")[1], ("Dm", "up") : wavfile.read("Audio/Dm[u].wav")[1],
        ("Em", "down") : wavfile.read("Audio/Em[d].wav")[1], ("Em", "up") : wavfile.read("Audio/Em[u].wav")[1],
        ("F", "down") : wavfile.read("Audio/F[d].wav")[1], ("F", "up") : wavfile.read("Audio/F[u].wav")[1],
        ("G", "down") : wavfile.read("Audio/G[d].wav")[1], ("G", "up") : wavfile.read("Audio/G[u].wav")[1]
    }
    
    # go through audio list, cut samples half a second long and multiply by volume
    samplerate = 44100
    half_samplerate = int(samplerate * 0.5)
    data = []
    for i in audio:
        data.append(np.float32(c_dir.get((i[0], i[2]))[:half_samplerate] * i[1]))

    # create output file
    data = np.concatenate(data, axis=0)
    wavfile.write("output.wav", samplerate, data.astype(np.float32))


if __name__ == "__main__":
    file_name = "Images/Aerosols_2025-08-19T20_00_00_7680x4320.png"
    img = load_image(file_name)
    img_s = slice_image(img)
    audio = analyser(img_s)
    create_wav(audio)
