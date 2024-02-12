from tensorflow.keras.models import load_model
import argparse
import pickle
import cv2 as cv

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-l", "--label-bin", required=True)
ap.add_argument("-w", "--width", type=int, default=28)
ap.add_argument("-e", "--height", type=int, default=28)
ap.add_argument("-f", "--flatten", type=int, default=-1)
args = vars(ap.parse_args())

image = cv.imread(args["image"])
output = image.copy()
image = cv.resize(image, (args["width"], args["height"]))

image = image.astype("float") / 255.0

if args["flatten"] > 0:
    image = image.flatten()
    image = image.reshape((1, image.shape[0]))
else:
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

preds = model.predict(image)

i = preds.argmax(axis = 1)[0]
label = lb.classes_[i]

text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv.putText(output, text, (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv.imshow("Image", output)
cv.waitKey(0)