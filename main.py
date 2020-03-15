from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
from mss import mss
import cv2
import numpy as np
from PIL import ImageGrab
import time
import pyautogui
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean
from directkeys import PressKey, ReleaseKey, W, A, S, D
import pyautogui

## Not to detect : 
## aeroplane
## bird
## boat
## cat
## chair
## diningtable
## pottedplant
## sheep
## sofa
## tvmonitor
## bottle
## motorbike

no_need = ['aeroplane', 'bird', 'boat', 'cat', 'chair', 'diningtable', 'pottedplant', 'sheep', 'sofa', 'tvmonitor', 'bottle', 'motorbike']

# utility functions
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    print('straight')

def left():
    PressKey(A)
    #ReleaseKey(W)#
    ReleaseKey(D)
    #ReleaseKey(A)   # Basically tapping the key 'A' once
    #ReleaseKey(W)
    print('left')

def right():
    PressKey(D)
    #ReleaseKey(W)#
    ReleaseKey(A)
    #ReleaseKey(D)   # Basically tapping the key 'A' once
    #ReleaseKey(W)
    print('right')

def stop():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def reposition(count):
    if count == 0:
        straight()
    while(count != 0):
        print(count)
        if count > 0:
            for i in range(3):
                straight()
            left()
            count -= 1
        elif count < 0:
            for i in range(3):
                straight()
            right()
            count += 1

net_type = r'mb1-ssd'
model_path = r'models/mobilenet-v1-ssd-mp-0_675.pth'
label_path = r'models/voc-model-labels.txt'

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

timer = Timer()

monitor = {'top' : 40, 'left' : 0, 'width' : 800, 'height' : 600}
sct = mss()

count = 0

## Show time
# delay
for i in range(4):
    print(4 - i)
    time.sleep(1)
# kick
for i in range(10):
    time.sleep(0.3)
    straight()

while True:
    sct_image = sct.grab(monitor)
    orig_image = np.array(sct_image)
    if orig_image is None:
        continue

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    #print("boxes : ", boxes)
    #print("labels : ", labels)
    #print("Probs : ", probs)
    interval = timer.end()
    #print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

    ## Game specific : Finding max probability of person, that'll be Jacob
    persons = []
    jacob = ['jacob', -1, [-1, -1, -1, -1]]   # characteristics of jacob
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = class_names[labels[i]]
        if str(label) == 'person':
            persons.append(float(probs[i]))
    #print('persons : ', persons)
    if(len(persons) != 0):
        jacob[1] = max(persons)     # person with highest probability will be jacob
        jacob[2] = [box[0], box[1], box[2], box[3]]
    #print('Jacob' , jacob)
    ##

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        if float(probs[i]) == jacob[1]:
            label = f"{jacob[0]}: {probs[i]:.2f}"
        label1 = label.split(':')[0]

        ## Game specific : overlapping obstacles with jacob
        if label1 not in no_need:
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

            cv2.putText(orig_image, label,
                        (box[0]+20, box[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type

            if str(label1) != 'jacob':
                if label1 == 'bicycle':     # cz in dark conditions jacob belt is detected as bicycle
                    if box[0] > (jacob[2][0] - 20) and box[0] < (jacob[2][0] + 20):
                        pass
                    elif box[2] > (jacob[2][2] - 20) and box[2] < (jacob[2][2] + 20):
                        pass
                elif (box[2] > (jacob[2][0])) and (box[2] < (jacob[2][2])):
                    right()
                    count += 1
                elif (box[0] > (jacob[2][0])) and (box[0] < (jacob[2][2])):
                    left()
                    count -= 1
                else:
                    #straight()
                    #print('straight')
                    # doesn't work in busy env cz while this executes, jacob collides with other objects
                    
                    reposition(count)
                    count = 0
         
        ##
        '''
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 2ww55, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
        '''
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
