import pandas as pd
import os
import ast
from sklearn import model_selection
from tqdm import tqdm
import numpy as np
import shutil

DATA_PATH=r"yolov5-master"
OUTPUT_PATH=r"yolov5-master\yolo_mask_data"

def process(data,data_type="train"):
    for _, row in tqdm(data.iterrows(),total=len(data)):
        image_name=row["filename"]
        bounding_boxes=row["bboxes"]
        yolo_data=[]
        for bbox in bounding_boxes:
            x=int(bbox[0])
            y=int(bbox[1])
            w=int(bbox[2])-x
            h=int(bbox[3])-y
            x_centre=(x+w)/2.0
            y_centre=(y+h)/2.0
            x_centre/=416.0
            y_centre/=416.0
            w/=416.0
            h/=416.0
            status=int(bbox[4])
            yolo_data.append([status,x_centre,y_centre,w,h])
        yolo_data=np.array(yolo_data)
        np.savetxt(
            os.path.join(OUTPUT_PATH , f"labels/{data_type}/{image_name}.txt"),
                         yolo_data,
                         fmt=["%d","%f","%f","%f","%f"]
            )
        shutil.copyfile(
            os.path.join(DATA_PATH,f"mask_data/train/{image_name}"),
            os.path.join(OUTPUT_PATH,f"images/{data_type}/{image_name}"),
            )
            
            
if __name__=="__main__":
   
    df=pd.read_csv(os.path.join(DATA_PATH, r"mask_data\train",r"_annotations.csv"))
    a = {'mask' : 1,'no-mask' : 0}
    df['status'] = df['status'].map(a)
    df["bbox"]="["+df["xmin"].astype(str)+" "+","+df["ymin"].astype(str)+" "+","+df["xmax"].astype(str)+" "+","+df["ymax"].astype(str)+" "+","+df["status"].astype(str)+"]"
    df.bbox=df.bbox.apply(ast.literal_eval)
    df=df[["filename","width","height","bbox","status"]]
    df=df.groupby("filename")["bbox"].apply(list).reset_index(name="bboxes")
    df_train,df_valid=model_selection.train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=True
        )
    df_train=df_train.reset_index(drop=True)
    df_valid=df_valid.reset_index(drop=True)
    
    process(df_train,data_type="train")
    process(df_valid,data_type="validation")
    