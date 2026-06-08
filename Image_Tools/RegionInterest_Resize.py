import cv2
import numpy as np
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from tensorflow.python.ops.numpy_ops import np_config
import tensorflow as tf
def ROI_IMG():
    img = cv2.imread('E:\\Tensorflow\\Resized\\img1.png')
    mask = np.zeros_like(img)
    roi_corners = np.array([[163, 192], [391, 208], [39, 592], [465, 563]], dtype=np.int32)
    color = (0, 255, 0) 
    cv2.rectangle(mask,roi_corners[0],roi_corners[3],color,-1)
    #cv2.fillPoly(mask, [roi_corners], (255,255,255))
    masked_img = cv2.bitwise_and(img, mask)
    cv2.imshow("Original Image", img)
    cv2.imwrite('maskeimg.jpg',mask)
    cv2.imwrite('maskecolorimg.jpg',masked_img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Masked Image", masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def RoiVedio():
    cap = cv2.VideoCapture('E:\\Tensorflow\\Helmet_video.mp4')
    roi_x, roi_y = 400, 300
    roi_w, roi_h = 900, 800
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        mask = np.zeros_like(frame)
        cv2.rectangle(mask, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2)
        masked_img = cv2.bitwise_and(frame, mask)
        #roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        #roi = cv2.GaussianBlur(roi, (5, 5), 0)
        cv2.imshow('Frame', masked_img)
        
        key = cv2.waitKey(25)
        if key == 27:
           cv2.imwrite("E:\\Tensorflow\\frame.jpg", masked_img)# Esc key
           break

    # Release the video file and close all windows
    cap.release()
    cv2.destroyAllWindows()

def vedioImg():
    roi_x, roi_y = 400, 300
    roi_w, roi_h = 900, 800
    cap = cv2.VideoCapture('E:\\Tensorflow\\Helmet_video.mp4')
    #mask1=[]
    #mask1[100:300, 200:400] = 255
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 25, (640, 480))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print("frame=",type(frame))
        #cv2.imshow('Frame1', frame)
        mask=np.zeros_like(frame)
        cv2.rectangle(mask, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255,0), -1)
        masked_frame = cv2.bitwise_and(frame, mask)
        out.write(masked_frame)
        print("masked frame=",type(masked_frame))
        cv2.imshow('Masked Video', masked_frame)
        if cv2.waitKey(1) == ord('q'):
            #frame1 = cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2RGB)
            cv2.imwrite("E:\\Tensorflow\\frame1.jpg", masked_frame)
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
def loadModel(path):
    PATH_TO_LABELS="E:\\Tensorflow\\Workspace\\Data\\label_map.pbtxt"
    print('Loading model...', end='')
    start_time = time.time()
    # Load saved model and build the detection function
    model = tf.saved_model.load(path)
    detect_fn = model.signatures['serving_default']

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
    return detect_fn,category_index

def snapshot():
    roi_x, roi_y = 400, 300
    roi_w, roi_h = 900, 800
    cap = cv2.VideoCapture('E:\\Tensorflow\\Helmet_video.mp4')
    i=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask=np.zeros_like(frame)
        cv2.rectangle(mask, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255,0), -1)
        masked_frame = cv2.bitwise_and(frame, mask)
        cv2.imshow('Masked Video', masked_frame)
        cv2.imwrite("E:\\Tensorflow\\frame"+str(i)+".jpg", masked_frame)
        i=i+1
        if cv2.waitKey(1) == ord('q'):
            #frame1 = cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2RGB)
            cv2.imwrite("E:\\Tensorflow\\frame1.jpg", masked_frame)
            break 

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def modelROI(detectFunction,lbl):
    roi_x, roi_y = 400, 300
    roi_w, roi_h = 900, 800
    cap = cv2.VideoCapture('E:\\Tensorflow\\outputFile.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #print("frame=",type(frame))
        mask=np.zeros_like(frame)
        cv2.rectangle(mask, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255,0), -1)
        masked_frame = cv2.bitwise_and(frame, mask)
        input_tensor = tf.convert_to_tensor(masked_frame)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detectFunction(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        image_np_with_detections =  masked_frame.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            lbl,
            use_normalized_coordinates=True,
            max_boxes_to_draw=1,
            min_score_thresh=.80,
            agnostic_mode=False)
        cv2.imshow("Detect", image_np_with_detections)
        if cv2.waitKey(1) == ord('q'):
            #frame1 = cv2.cvtColor(masked_frame, cv2.COLOR_GRAY2RGB)
            cv2.imwrite("E:\\Tensorflow\\frame1.jpg", masked_frame)
            break

    cap.release()
    cv2.destroyAllWindows()

def videomodel():

    #roi_x, roi_y = 400, 300
    #roi_w, roi_h = 900, 800
    roi_x, roi_y = 155, 251
    roi_w, roi_h = 320, 460
    path="E:\\Tensorflow\\Workspace\\SLN\\TFOD\\H171\\cropped09"
    # Load the TensorFlow detection model
    model_path="E:\Tensorflow\Workspace\models\Model319\Trained\saved_model"
    #model_path = "E:\\Tensorflow\\Workspace\\models\\faster_rcnn_resnet101_v1_640x640_coco17_tpu-8\\Trained Model\\saved_model"
    model = tf.saved_model.load(model_path)
    # Open the video file
    video_path = "E:\\Tensorflow\\cropped.mp4"
    cap = cv2.VideoCapture(video_path)
    confidence_threshold=.60
    c=0
    c1=0
    dim=(640,640)
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            resized = cv2.resize(frame, dim, interpolation = cv2.INTER_LINEAR)
            #cv2.imwrite("resized.jpg" ,resized)
            mask=np.zeros_like(resized)
            cv2.rectangle(mask, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255,0), -1)
            masked_frame = cv2.bitwise_and(resized, mask)
            cv2.imwrite("mask"+str(c1)+".jpg" ,masked_frame)
            c1=c1+1
            # Convert the frame to a tensor and preprocess it
            tensor = tf.convert_to_tensor(masked_frame)
            #tensor = tf.image.resize(tensor, (input_height, input_width))
            tensor = tensor[np.newaxis, ...]

            # Run the detection model on the frame
            detections = model(tensor)
            boxes = detections['detection_boxes'][0].numpy()
            scores = detections['detection_scores'][0].numpy()
            classes = detections['detection_classes'][0].numpy()
            print(type(boxes))
            print(type(scores))
            print(type(classes))
            # Draw the bounding boxes on the frame
            #for box, label, score in zip(detections['detection_boxes'][0], detections['detection_classes'][0], detections['detection_scores'][0]):
            #    #print(type(box))
            #    if score > confidence_threshold:
            #        ymin, xmin, ymax, xmax = box.numpy()
            #        xmin = int(xmin * frame.shape[1])
            #        xmax = int(xmax * frame.shape[1])
            #        ymin = int(ymin * frame.shape[0])
            #        ymax = int(ymax * frame.shape[0])
            #        cv2.rectangle(masked_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    #cv2.putText(frame, label_map[label.numpy()], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #print(type(scores))
            #print(len(boxes))
            for i in range(len(boxes)):
               
                if scores[i] >  confidence_threshold:
                    ymin, xmin, ymax, xmax = boxes[i]
                    cv2.rectangle(masked_frame, (int(xmin*640), int(ymin*640)), (int(xmax*640), int(ymax*640)), (0, 255, 0), 2)
                    cv2.putText(masked_frame, f'{classes[i]}: {scores[i]:.2f}', (int(xmin*640), int(ymin*640)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Display the frame
                    cv2.imwrite(path +str(c)+ ".jpg" ,masked_frame)
                    c=c+1
            #cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video file and close the display window
    cap.release()
    cv2.destroyAllWindows()
def ImgModel(path,img):

    image = cv2.imread(img)
    #image = np.array(image, dtype=np.float32)
    #image = np.expand_dims(image, axis=0)
    tensor=np.array(image)
    tensor = tf.convert_to_tensor(image)
    tensor = tensor[np.newaxis, ...]
    model = tf.saved_model.load(path)

    # Load the pre-trained weights
    #model.load_weights('path/to/weights')

    # Run the model on the image
    detections = model(tensor)

    # Postprocess the predictions
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy()

    # Display the results
    for i in range(len(boxes)):
        if scores[i] > 0.50:
            ymin, xmin, ymax, xmax = boxes[i]
            cv2.rectangle(image, (int(xmin*640), int(ymin*640)), (int(xmax*640), int(ymax*640)), (0, 255, 0), 2)
            cv2.putText(image, f'{classes[i]}: {scores[i]:.2f}', (int(xmin*640), int(ymin*640)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Image', image)
    cv2.imwrite("imgdetect.jpg",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detectHT(path):
    object_classes = ['helmet', 'turban']
    model = tf.saved_model.load(path)
    min_confidence = 0.80
    video_path = "E:\\Tensorflow\\outputFile1.mp4"
    cap = cv2.VideoCapture(video_path)
    infer = model.signatures['serving_default']
    c1=0
    dim=(640,640)
    roi_x, roi_y = 155, 251
    roi_w, roi_h = 450, 550
    path1="E:\\Tensorflow\\Workspace\\SLN\\TFOD\\H171\\turban09"
    detection=[]
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if ret:
            resized = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
            #cv2.imwrite("resized.jpg" ,resized)
            mask = np.zeros_like(resized)
            cv2.rectangle(mask, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), -1)
            masked_frame = cv2.bitwise_and(resized, mask)
            # Normalize the image pixels
            #normalized_image = resized_image / 255.0
            #cv2.imwrite("mask"+str(c1)+".jpg" ,masked_frame)
            #c1=c1+1
            # Convert the frame to a tensor and preprocess it
            #input_image = np.expand_dims(normalized_image, axis=0)

            tensor = tf.convert_to_tensor(masked_frame)
            #tensor = tf.image.resize(tensor, (input_height, input_width))
            tensor = tensor[np.newaxis, ...]
            #image_tensor = tf.expand_dims(image_tensor, 0)
            # Make predictions on the input image
            predictions = model(tensor)
            print("D1",detection)
            # Loop over the detected objects
            detections = predictions['detection_boxes'][0].numpy()
            # Filter the detections by confidence and class
            filtered_detections = []
            for i in range(predictions['detection_scores'][0].shape[0]):
                class_id = int(predictions['detection_classes'][0][i])
                if class_id < len(object_classes) and predictions['detection_scores'][0][i] >= min_confidence:
                    bbox = detections[i] * np.array([masked_frame.shape[0], masked_frame.shape[1], masked_frame.shape[0], masked_frame.shape[1]])
                    bbox = bbox.astype(np.int32)
                    filtered_detections.append((object_classes[class_id], predictions['detection_scores'][0][i].numpy(), bbox))
            print("D2",detection)
            # Draw the filtered detections on the image
            for detection in filtered_detections:
                class_name, confidence, bbox = detection
                # ymin, xmin, ymax, xmax = bbox[detection]

                cv2.rectangle(masked_frame, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0), 2)
                #cv2.rectangle(masked_frame, (int(xmin*640), int(ymin*640)), (int(xmax*640), int(ymax*640)), (0, 255, 0), 2)
                cv2.putText(masked_frame, '{}: {:.2f}'.format(class_name, confidence), (bbox[1], bbox[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(path1+str(c1)+".jpg" ,masked_frame)
                c1=c1+1


PATH_TO_SAVED_MODEL = "E:\\Tensorflow\\Workspace\\models\\faster_rcnn_resnet101_v1_640x640_coco17_tpu-8\\Trained Model\\saved_model"
p="E:\\Tensorflow\\Workspace\\Images\\test\\img70.png"
#ImgModel(PATH_TO_SAVED_MODEL,p)
#detectHT("E:\Tensorflow\Workspace\models\Model319\Trained\saved_model")
videomodel()
#detectFunction,lbl=loadModel(PATH_TO_SAVED_MODEL)
#modelROI(detectFunction,lbl)
#vedioImg()
#RoiVedio()
#snapshot()
#ROI_IMG()
