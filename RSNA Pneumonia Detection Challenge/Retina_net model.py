#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import scipy.misc
import pydicom 
import glob
import sys
import os
import pandas as pd 
import base64
from IPython.display import HTML


# In[ ]:


from scipy.ndimage.interpolation import zoom


# In[ ]:


get_ipython().system('git clone https://github.com/fizyr/keras-retinanet')
os.chdir("keras-retinanet") 
get_ipython().system('python setup.py build_ext --inplace')


# In[ ]:


DATA_DIR = "/kaggle/input/"
ROOT_DIR = "/kaggle/working/"


# In[ ]:


train_pngs_dir = os.path.join(DATA_DIR, "rsna-pneu-train-png/stage_1_train_pngs/orig/")
test_dicoms_dir  = os.path.join(DATA_DIR, "rsna-pneumonia-detection-challenge/stage_1_test_images/") 


# In[ ]:


bbox_info = pd.read_csv(os.path.join(DATA_DIR, "rsna-pneumonia-detection-challenge/stage_1_train_labels.csv"))
detailed_class_info = pd.read_csv(os.path.join(DATA_DIR, "rsna-pneumonia-detection-challenge/stage_1_detailed_class_info.csv"))
detailed_class_info = detailed_class_info.drop_duplicates()


# In[ ]:


positives = detailed_class_info
positives.head()


# In[ ]:


cash_class = positives
cash_class.head()


# In[ ]:


positives = positives.merge(bbox_info, on="patientId")
positives = positives[["patientId", "x", "y", "width", "height"]]
positives = positives.merge(cash_class, on="patientId")


# In[ ]:


#positives[positives["class"]=='No Lung Opacity / Not Normal']

#positives["class"] = np.where(positives["class"]=='No Lung Opacity / Not Normal', 'Normal','')


conditions = [
         positives["class"]=='No Lung Opacity / Not Normal',
         positives["class"]=='Normal',
         positives["class"]=='Lung Opacity'
             ]
choices = ['','','Lung Opacity']
positives["class"] = np.select(conditions, choices)
positives.head()


# In[ ]:



positives["patientId"] = [os.path.join(train_pngs_dir, "{}.png".format(_)) for _ in positives.patientId]
positives["x1"] = positives["x"] 
positives["y1"] = positives["y"] 
positives["x2"] = positives["x"] + positives["width"]
positives["y2"] = positives["y"] + positives["height"]
positives["Target"] = positives["class"]
del positives["x"], positives["y"], positives["width"], positives["height"]


# In[ ]:


del positives['class']


# In[ ]:


positives.head()


# In[ ]:


annotations = positives


# In[ ]:


annotations = annotations.fillna(88888)
annotations["x1"] = annotations.x1.astype("int32").astype("str") 
annotations["y1"] = annotations.y1.astype("int32").astype("str") 
annotations["x2"] = annotations.x2.astype("int32").astype("str") 
annotations["y2"] = annotations.y2.astype("int32").astype("str")
annotations = annotations.replace({"88888": ""}) 
annotations = annotations[["patientId", "x1", "y1", "x2", "y2", "Target"]]
annotations.to_csv(os.path.join(ROOT_DIR, "annotations.csv"), index=False, header=False)


# In[ ]:


annotations.head()


# In[ ]:


classes_file = pd.DataFrame({"class": ["Lung Opacity"], "id": [0]}) 
classes_file.to_csv(os.path.join(ROOT_DIR, "classes.csv"), index=False, header=False)


# In[ ]:


classes_file.head()


# In[ ]:


annotations.shape


# In[ ]:


get_ipython().system('python /kaggle/working/keras-retinanet/keras_retinanet/bin/train.py --backbone "resnet50" --image-min-side 608 --image-max-side 608 --batch-size 8  --epochs 1 --steps 3623 --no-snapshots csv /kaggle/working/annotations.csv /kaggle/working/classes.csv')


# In[ ]:


model.save("model.h5")


# In[ ]:


get_ipython().system('python /kaggle/working/keras-retinanet/keras_retinanet/bin/convert_model.py /kaggle/working/keras-retinanet/snapshots/resnet50_csv_1.h5 /kaggle/working/keras-retinanet/converted_model.h5 ')


# In[ ]:


from keras_retinanet.models import load_model 
retinanet = load_model(os.path.join(ROOT_DIR, "keras-retinanet/converted_model.h5"), 
                       backbone_name="resnet50")


# In[ ]:


def preprocess_input(x):
    x = x.astype("float32")
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.680
    return x


# In[ ]:


test_dicoms = glob.glob(os.path.join(test_dicoms_dir, "*.dcm"))
test_patient_ids = [_.split("/")[-1].split(".")[0] for _ in test_dicoms]
test_predictions = [] 
for i, dcm_file in enumerate(test_dicoms): 
    sys.stdout.write("Predicting images: {}/{} ...\r".format(i+1, len(test_dicoms)))
    sys.stdout.flush() 
    # Load DICOM and extract pixel array 
    dcm = pydicom.read_file(dcm_file)
    arr = dcm.pixel_array
    # Make 3-channel image
    img = np.zeros((arr.shape[0], arr.shape[1], 3))
    for channel in range(img.shape[-1]):
        img[..., channel] = arr 
    # Resize 
    # Change image size if necessary!
    scale_factor = 256. / img.shape[0]
    img = zoom(img, [scale_factor, scale_factor, 1], order=1, prefilter=False)
    # Preprocess with ImageNet mean subtraction
    img = preprocess_input(img) 
    prediction = retinanet.predict_on_batch(np.expand_dims(img, axis=0))
    test_predictions.append(prediction) 


# In[ ]:


test_pred_df = pd.DataFrame() 
for i, pred in enumerate(test_predictions):
    # Take top 5 
    # Should already be sorted in descending order by score
    bboxes = pred[0][0][:5]
    scores = pred[1][0][:5]
    # -1 will be predicted if nothing is detected
    detected = scores > -1 
    if np.sum(detected) == 0: 
        continue
    else:
        bboxes = bboxes[detected]
        bboxes = [box / scale_factor for box in bboxes]
        scores = scores[detected]
    individual_pred_df = pd.DataFrame() 
    for j, each_box in enumerate(bboxes): 
        # RetinaNet output is [x1, y1, x2, y2] 
        tmp_df = pd.DataFrame({"patientId": [test_patient_ids[i]], 
                               "x": [each_box[0]],  
                               "y": [each_box[1]], 
                               "w": [each_box[2]-each_box[0]],
                               "h": [each_box[3]-each_box[1]],
                               "score": [scores[j]]})
        individual_pred_df = individual_pred_df.append(tmp_df) 
    test_pred_df = test_pred_df.append(individual_pred_df) 

test_pred_df.head()


# In[ ]:


threshold = 0.50

list_of_pids = [] 
list_of_preds = [] 
for pid in np.unique(test_pred_df.patientId): 
    tmp_df = test_pred_df[test_pred_df.patientId == pid]
    tmp_df = tmp_df[tmp_df.score >= threshold]
    # Skip if empty
    if len(tmp_df) == 0:
        continue
    predictionString = " ".join(["{} {} {} {} {}".format(row.score, row.x, row.y, row.w, row.h) for rownum, row in tmp_df.iterrows()])
    list_of_preds.append(predictionString)
    list_of_pids.append(pid) 

positives = pd.DataFrame({"patientId": list_of_pids, 
                          "PredictionString": list_of_preds}) 

negatives = pd.DataFrame({"patientId": list(set(test_patient_ids) - set(list_of_pids)), 
                          "PredictionString": [""] * (len(test_patient_ids)-len(list_of_pids))})

submission = positives.append(negatives)


# In[ ]:


def create_download_link(df, title = "Download CSV file", filename = "RSNA_DataSet.csv"):  
    csv = df.to_csv(index = 0)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# In[ ]:


test_pred_df.head(100)

