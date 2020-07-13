import gbp
import wave
import whgo
#import hog
import time
import cv2
import centrist
import numpy as np
import os

directory = "/media/neduchal/data2/datasety/miniplaces/images"

train_names = open("filelist.txt", "r").read().split("\n")
if len(train_names[-1]) == 0:
    train_names = train_names[:-1]
test_names = train_names[0:100]
gbp_times = []
centrist_times = []
whgo_times = []
wave_times = []
cl = centrist.load()
for t in test_names:
    fn = os.path.join(directory, t[7:])
    img = cv2.imread(fn)
    # print(fn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # GBP
    start = time.time()
    gbp.gbp(gray)
    gbp_times.append(time.time() - start)

    # CENTRIST
    start = time.time()
    centrist.centrist_im(cl, gray)
    centrist_times.append(time.time() - start)

    # WHGO
    start = time.time()
    whgo.whgo(gray, 16)
    whgo_times.append(time.time() - start)

    # WAVE
    start = time.time()
    wave.wave(gray)
    wave_times.append(time.time() - start)

print("GBP time", np.mean(gbp_times))
print("CENTRIST time", np.mean(centrist_times))
print("WHGO time", np.mean(whgo_times))
print("WAVE time", np.mean(wave_times))
