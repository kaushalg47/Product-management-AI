
import math
import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model

from roboflow import Roboflow
rf = Roboflow(api_key="cfdKq2OHtlZZFRxJqtV6")
project = rf.workspace().project("sku-110k")
model = project.version(4).model

# infer on a local image
results = model.predict(r"C:\Users\kaushal\Downloads\sample3.jpg", confidence=40, overlap=30).json()
print(results['predictions'])

def extractObjectList(input_folder):
    obj_list = []
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            print(f"Processing: {filename}")
            obj_list.append(filename)
    return obj_list            

def extractFeatureList(input_folder):
    feature_list = []
    obj_list = []

    base_model = InceptionV3(weights='imagenet', include_top=False)

    feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed7').output)

    def extract_flattened_features(im_path):
        img = image.load_img(im_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = feature_extractor.predict(img_array)

        flattened_features = features.flatten()

        return flattened_features

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            print(f"Processing: {filename}")
            obj_list.append(filename)
            feature_list.append(extract_flattened_features(input_path))

    print("Done!")
    return feature_list       

def classification(image_path, results, features_list, obj_list):
    image_real = Image.open(image_path)

    xywh_values = []
    labels1 = []

    # Load a pre-trained InceptionV3 model (you can choose a different model if needed)
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # Create a model that extracts features from the intermediate layers
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed7').output)

    def cosine_similarity(v1, v2):
        "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]; y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)

    def extract_flattened_features(img):
        img = img.resize((299, 299))
        img_array = image.img_to_array(img) 
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = feature_extractor.predict(img_array)

        flattened_features = features.flatten()

        return flattened_features
    def append_to_limit_group(limits_dict, live_data_name, live_data_number, low, high):
        for limits, name_list in limits_dict.items():
            lower_limit, upper_limit = limits
            if lower_limit <= live_data_number <= upper_limit:
                name_list.append(live_data_name)
                return 

            
        new_lower_limit = low  
        new_upper_limit = high
        new_key = (new_lower_limit, new_upper_limit)
        limits_dict[new_key] = [live_data_name]



    names =[]
    main = []
    limit_dict = {}
    flag = 0
    full_img = cv2.imread(image_path) 
    for i, prediction in enumerate(results['predictions']):
        x, y, w, h, name = prediction['x'], prediction['y'], prediction['width'], prediction['height'], prediction['class']
        xywh_values.append((x, y, w, h))
        labels1.append(name)

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        color = (0, 255, 0) 
        thickness = 2
        cv2.rectangle(full_img, (x1, y1), (x2, y2), color, thickness)

        # Crop using the Image.crop method
        window = image_real.crop((x1, y1, x2, y2))
        image_array = np.array(window)
        window_pil = Image.fromarray(image_array)

        ext_feature = extract_flattened_features(window_pil)
        simis = []
        for exist_features in features_list:
            simis.append(cosine_similarity(exist_features, ext_feature))
        max_similarity = max(simis)
        max_similarity_index = simis.index(max_similarity)
        matched_name = obj_list[max_similarity_index]
        names.append(matched_name)
        print(y1,y2,y,matched_name)
        
        #rack-logic
        append_to_limit_group(limit_dict, matched_name, y, y1,y2)
        

        label_position = (x1, y1 - 10) 
        cv2.putText(full_img, matched_name, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    print(limit_dict)
    output_image_path = r"C:\Users\kaushal\Downloads\output_image_with_boxes.jpg"
    cv2.imwrite(output_image_path, full_img)
    print(f"Image with bounding boxes and labels saved to: {output_image_path}")

    return main

input_folder = r"C:\Users\kaushal\Downloads\innov_hack\new\objects"
image_path = r"C:\Users\kaushal\Downloads\sample3.jpg"
print(classification(image_path, results, extractFeatureList(input_folder), extractObjectList(input_folder)))
