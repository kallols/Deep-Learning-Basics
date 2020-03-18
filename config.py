# import the necessary packages
import os

# initialize the list of class label names
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog",
           "frog", "horse", "ship", "truck"]

# define the minimum learning rate, maximum learning rate, batch size,
# step size, CLR method, and number of epochs
MIN_LR = 1e-5
MAX_LR = 1e-2
BATCH_SIZE = 64
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 48

# define the path to the output training history plot and cyclical
# learning rate plot
LRFIND_PLOT_PATH = os.path.sep.join(["/home/oto/PycharmProjects/deep_learning", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["/home/oto/PycharmProjects/deep_learning", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["/home/oto/PycharmProjects/deep_learning", "clr_plot.png"])
