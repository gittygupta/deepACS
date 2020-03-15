from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
from mss import mss
import cv2
import numpy as np

## Not to detect : 
## aeroplanes
## bird
## boat
## cat
## chair
## diningtable
## pottedplant
## sheep
## sofa
## tvmonitor

no_need = ['aeroplanes', 'bird', 'boat', 'cat', 'chair', 'diningtable', 'pottedplant', 'sheep', 'sofa', 'tvmonitor']

net_type = 'mb1-ssd'
model_path = 'models/mobilenet-v1-ssd-mp-0_675.pth'
label_path = 'models/voc-model-labels.txt'

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(model_path)
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

timer = Timer()

monitor = {'top' : 40, 'left' : 0, 'width' : 800, 'height' : 600}
sct = mss()

while True:
    sct_image = sct.grab(monitor)
    orig_image = np.array(sct_image)
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        label1 = label.split(':')[0]
        if label1 not in no_need:
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

            cv2.putText(orig_image, label,
                        (box[0]+20, box[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
