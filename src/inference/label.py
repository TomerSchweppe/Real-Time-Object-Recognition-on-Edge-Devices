#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import threading
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
import time
plt.rcdefaults()

# PARAMETERS
GRID = 3 # 3x3 GRID
THRESHOLD = 0.9 # Threshold for detection
NUM_FRAMES = 60 # Sample fps
MARGIN = 20 # Margin for text
WEBCAM_CHANNEL = 1 # (0|1)
SAMPLE = False
NAMING = ['NW','N','NE','W','C','E','SW','S','SE']

import rospy
from geometry_msgs.msg import Twist
class Robot():
    def __init__(self):

        rospy.init_node('GoInCircle', anonymous=False)
        rospy.loginfo("To stop TurtleBot CTRL + C")
        rospy.on_shutdown(self.shutdown)


        self._cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist,
        queue_size=10)

        self._r = rospy.Rate(10)
        self._move_cmd = Twist()
        #r.sleep()

    def move(self, direction):
        if direction == 1: # NW
            self._move_cmd.linear.x = 0.0
            self._move_cmd.angular.z = 1.0
        elif direction == 2: # N
            self._move_cmd.linear.x = 1.0
            self._move_cmd.angular.z = 0.0
        elif direction == 3: # NE
            self._move_cmd.linear.x = 0.0
            self._move_cmd.angular.z = 1.0
        elif direction == 4: # W
            self._move_cmd.linear.x = 0.0
            self._move_cmd.angular.z = 1.0
        elif direction == 5: # C
            self._move_cmd.linear.x = 1.0
            self._move_cmd.angular.z = 0.0
        elif direction == 6: # E
            self._move_cmd.linear.x = 0.0
            self._move_cmd.angular.z = 1.0
        elif direction == 7: # SW
            self._move_cmd.linear.x = 0.0
            self._move_cmd.angular.z = 1.0
        elif direction == 8: # S
            self._move_cmd.linear.x = 1.0
            self._move_cmd.angular.z = 0.0
        elif direction == 9: # SE
            self._move_cmd.linear.x = 0.0
            self._move_cmd.angular.z = 1.0

        self._cmd_vel.publish(self._move_cmd)

    def search(self):
        pass

    def shutdown(self):
        rospy.loginfo("Stop TurtleBot")
        print('end')

        self._cmd_vel.publish(Twist())

        rospy.sleep(1)

def live_plotter(values):
    """
    Dynamic bar plot
    values: network confidence values for the full image and each box
    """
    plt.ion()
    plt.show()
    plt.figure(num=1, figsize=(10, 10))
    objects = ['Conf']
    for i in range(GRID * GRID):
        objects.append(NAMING[i])
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, values, align='center', alpha=0.5, color=['black'])
    plt.axhline(y=THRESHOLD, color='red')
    plt.xticks(y_pos, objects)
    plt.yticks(np.arange(0, 1 + 1, 0.1))
    plt.ylabel('Score')
    plt.title('Detection Threshold: ' + str(THRESHOLD))
    plt.ylim(0, 1)
    plt.pause(0.001)
    plt.clf()

def detect(frame,sess):
    """
    Detect object in frame
    """
    # input_image = frame
    # height, width, channels = input_image.shape
    # blockHeight = height // GRID
    # blockWidth = width // GRID
    #
    # batch = []
    #
    # # rows
    # for r in range(1,GRID+1):
    #     mask = input_image.copy()
    #     mask[:blockHeight*(r-1), :, :] = 0
    #     mask[blockHeight*r:,:,:] = 0
    #     batch.append(preprocess(mask,input_height,input_width))
    #
    # # cols
    # for c in range(1,GRID+1):
    #     mask = input_image.copy()
    #     mask[:, :blockWidth*(c-1), :] = 0
    #     mask[:, blockWidth*c:, :] = 0
    #     batch.append(preprocess(mask, input_height, input_width))
    #
    # # batch inference
    # batch = inference(np.concatenate(batch),sess)
    #
    # # grade each box
    # one_probs = []
    # for i in range(GRID*GRID):
    #     one_probs.append(batch[i//3]*batch[i%3+GRID])
    #
    # # drawing
    # boxes(frame)
    #
    # mark(input_image, np.argmax(one_probs) + 1, height, width)
    #
    # return one_probs,input_image
    """
     Detect object in frame
     """
    inputImage = frame
    height, width, channels = inputImage.shape
    blockHeight = height // GRID
    blockWidth = width // GRID

    batch = []
    for i in range(GRID * GRID):
        mask = inputImage.copy()
        mask = mask[blockHeight * (i // GRID):min(blockHeight * (i // GRID + 1), height - 1),
               blockWidth * (i % GRID):min(blockWidth * (i % GRID + 1), width - 1), :]
        batch.append(preprocess(mask, input_height, input_width))

    one_probs = inference(np.concatenate(batch), sess)

    boxes(inputImage)

    mark(inputImage, np.argmax(one_probs) + 1, height, width)

    return (np.argmax(one_probs) + 1,one_probs.tolist(), inputImage)

def boxes(image):
    """
    Draw boxes on image
    horizontal lines, vertical lines and text for each block
    """
    height, width, _ = image.shape
    blockHeight = height // GRID
    blockWidth = width // GRID
    # rows
    for i in range(1,GRID):
        image[blockHeight*i-1,:,:] = 0
    # cols
    for i in range(1,GRID):
        image[:,blockWidth*i-1,:] = 0
    boxNum = 0

    for i in range(0 + MARGIN, GRID * blockHeight, blockHeight):
        for j in range(0 + MARGIN, GRID * blockWidth, blockWidth):
            cv2.putText(image, NAMING[boxNum], (j, i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            boxNum = boxNum + 1

def mark(image,box,height,width):
    """
    Mark given box  
    """
    channels = 3
    blockHeight = height // GRID
    blockWidth = width // GRID
    col = box % GRID
    row = (box-1) // GRID + 1
    if col == 0:
        col = col + GRID

    for i in range(channels):
        value = (i==1)*255
        image[(row-1)*blockHeight:min(row*blockHeight,height-1),(col - 1) * blockWidth,i] = value
        image[(row-1)*blockHeight:min(row*blockHeight,height-1),min(col * blockWidth,width-1),i] = value
        image[(row-1)*blockHeight,(col - 1) * blockWidth:min(col * blockWidth,width-1),i] = value
        image[min(row * blockHeight,height-1),(col - 1) * blockWidth:min(col * blockWidth,width-1),i] = value
    cv2.putText(image, NAMING[box-1], ((col-1)*blockWidth+MARGIN, (row-1)*blockHeight+MARGIN), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def load_graph(model_file):
    """
    Load input graph
    """
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

def preprocess(frame,
                input_height=299,
                input_width=299,
                input_mean=0,
                input_std=255):
    """
    Preprocess frame
    """
    float_caster = frame.astype(float)
    resized = cv2.resize(float_caster, dsize=(input_width,input_height))
    normalized = np.divide(np.subtract(resized, input_mean), input_std)
    result = np.expand_dims(normalized,axis=0)
    return result

def load_labels(label_file):
    """
    Load labels
    """
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def inference(input,sess):
    """
    Inference 
    """
    results = sess.run(output_operation.outputs[0], {
    input_operation.outputs[0]: input
    })
    return results[:,1]

def preprocess_inference(frame,sess):
    """
    Inference and preprocess 
    """
    t = preprocess(
        frame,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)
    return inference(t, sess)

def image(frame):
    """
    Run on one image
    """
    cv2.imwrite('img.jpg',frame)
    with tf.Session(graph=graph) as sess:
        _, one_probs, frame = detect(frame, sess)
        live_plotter(one_probs)
        cv2.imshow("Live", frame)
    cv2.waitKey(0) # exit on ESC

def webcam():
    """
    Run on webcam
    """
    robot = Robot()
    cv2.namedWindow("Live")
    vc = cv2.VideoCapture(WEBCAM_CHANNEL)
    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        while rval:
            start = time.time()
            for f in range(NUM_FRAMES):
                if SAMPLE and f%2==0: # skip second frame
                    cv2.imshow("Live", frame)
                    continue
                one_prob = preprocess_inference(frame,sess)
                if one_prob > THRESHOLD:
                    direction, one_probs,frame = detect(frame,sess)
                    robot.move(direction)
                else:
                    boxes(frame)
                    robot.search()
                cv2.imshow("Live",frame)


                key = cv2.waitKey(20)
                if key == 32: # space bar
                    cv2.imshow("Sample",frame)
                    if one_prob > THRESHOLD:
                        live_plotter([one_prob]+one_probs)
                    else:
                        live_plotter([one_prob] + [0]*(GRID*GRID))

                rval, frame = vc.read()

                if key == 27 or rval == False:  # exit on ESC
                    rval = False
                    robot.shutdown()
                    break
            else:  # no break
                print(str(round(NUM_FRAMES / (time.time() - start), 2)) + ' FRAMES PER SECOND')


if __name__ == "__main__":
    file_name = None
    model_file = None
    label_file = None
    input_height = None
    input_width = None
    input_mean = 0
    input_std = 255
    input_layer = None
    output_layer = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--graph", help="graph/model to be executed",default='./graph.pb')
    parser.add_argument("--labels", help="name of file containing labels",default='./output_labels.txt')
    parser.add_argument("--input_height", type=int, help="input height",default=224)
    parser.add_argument("--input_width", type=int, help="input width",default=224)
    parser.add_argument("--input_mean", type=int, help="input mean")
    parser.add_argument("--input_std", type=int, help="input std")
    parser.add_argument("--input_layer", help="name of input layer",default='Placeholder')
    parser.add_argument("--output_layer", help="name of output layer",default='final_result')
    args = parser.parse_args()

    if args.graph:
        model_file = args.graph
    if args.image:
        file_name = args.image
    if args.labels:
        label_file = args.labels
    if args.input_height:
        input_height = args.input_height
    if args.input_width:
        input_width = args.input_width
    if args.input_mean:
        input_mean = args.input_mean
    if args.input_std:
        input_std = args.input_std
    if args.input_layer:
        input_layer = args.input_layer
    if args.output_layer:
        output_layer = args.output_layer


    graph = load_graph(model_file)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    if file_name is not None:
        image(cv2.imread(file_name))
    else: # use webcam
        webcam()
    cv2.destroyWindow("preview") # cleanup

