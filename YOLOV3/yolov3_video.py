'''
In this file we are going to give image to yolov3 pretrained model
'''
import numpy as np
import cv2
threashold = 0.6
image_size = 320
def best_boxes(total_outputs):
    best_bounding_boxes = []
    best_confidence = []
    best_class_index = []

    for i in total_outputs:
        for j in i:
            class_pro_values = j[5 : ]
            high_class_index = np.argmax(class_pro_values)
            confidence = class_pro_values[high_class_index]
            if confidence > threashold:
                w, h = int(j[2] * image_size), int(j[3] * image_size)
                x, y = int(j[0] * image_size - w / 2), int(j[1] * image_size - h / 2)
                best_bounding_boxes.append([x,y,w,h])
                best_class_index.append(high_class_index)
                best_confidence.append(confidence)

    print(best_bounding_boxes)
    print(best_confidence)
    print(best_class_index)
    final_box = cv2.dnn.NMSBoxes(best_bounding_boxes,best_confidence,threashold,0.5)
    print("=====================")
    print(final_box)
    return best_bounding_boxes,best_confidence,best_class_index,final_box

def final_prediction(pixel_values,all_box,all_acc,all_index,final_box,height_ratio,width_ration):
    for p in final_box:
        x,y,w,h = all_box[p]
        x = int(x * width_ration)
        y = int(y * height_ratio)
        w = int(w * width_ration)
        h = int(h * height_ratio)
        acc_value = round(all_acc[p],2)
        class_name = all_class_names[all_index[p]]
        text_image = class_name +" "+ str(acc_value)
        font_stype = cv2.FONT_HERSHEY_PLAIN
        cv2.rectangle(pixel_values,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(pixel_values,text_image,(x,y-3),font_stype,1,(0,0,0),2)



image = cv2.VideoCapture("yolo_test.mp4")
# print(f"Original_image_shape : {image.shape}") # (450,600,3)
original_h , original_w = image.get(4) , image.get(3)


all_class_names = []
with open("class_names","r") as t:
    for i in t.readlines():
        all_class_names.append(i.strip())


#neural_network = cv2.dnn.readNetFromDarknet("yolov3 (1).cfg","yolov3.weights")   #yolo v3
neural_network = cv2.dnn.readNetFromDarknet("yolov4.cfg","yolov4.weights")  # yolo v4
# neural_network is a varibale we have architecure and weights
all_layer_names = neural_network.getLayerNames()  # all architecure layer names
important_layers = neural_network.getUnconnectedOutLayersNames() # ('yolo_82', 'yolo_94', 'yolo_106')
important_layers_index = neural_network.getUnconnectedOutLayers() # (200 , 227 , 254)
output_layers = [all_layer_names[k-1] for k in neural_network.getUnconnectedOutLayers()]

while True:
    res,pixel_values = image.read()
    if res == True:
        input_image = cv2.dnn.blobFromImage(pixel_values, 1 / 255, (320, 320), True, crop=False)
        # print(f"Input Image to Model : {input_image.shape}") # (1,3,320,320)
        # sent input to yolov3 architecure
        neural_network.setInput(input_image)

        total_outputs = neural_network.forward(output_layers)
        all_boxes,all_acc,all_index,final_box = best_boxes(total_outputs)

        final_prediction(pixel_values,all_boxes,all_acc,all_index,final_box,original_h/320,original_w/320)

        cv2.imshow("Current Frame ",pixel_values)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

image.release()
cv2.destroyAllWindows()