# USAGE
# python predict.py --image images/dog.jpg --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
# python predict.py --image images/dog.jpg --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --width 64 --height 64

# import the necessary packages
from keras.models import load_model
import pickle
import cv2

# load the input image and resize it to the target spatial dimensions
image = '/home/oto/Downloads/keras-tutorial/images/dog.jpg'
image = cv2.imread(image)
output = image.copy()
image = cv2.resize(image, (64, 64))

# scale the pixel values to [0, 1]
image = image.astype("float") / 255.0

# check to see if we should flatten the image and add a batch
# dimension
flatten = 0
if flatten > 0:
    image = image.flatten()
    image = image.reshape((1, image.shape[0]))

# otherwise, we must be working with a CNN -- don't flatten the
# image, simply add the batch dimension
else:
    image = image.reshape((1, image.shape[0], image.shape[1],
                           image.shape[2]))
# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model("/home/oto/PycharmProjects/deep_learning/small_vgg_model.h5")

# make a prediction on the image
preds = model.predict(image)

# find the class label index with the largest corresponding
# probability
lb = pickle.loads(open("label_bin", "rb").read())
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]
print("predicted image is: " + str(label))
# draw the class label + probability on the output image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255), 2)

# show the output image
cv2.imshow("Image", output)
cv2.waitKey(0)
