import cv2, numpy as np
import tensorflow as tf
import keras
import matplotlib as mpl
import pandas as pd
from glob import glob
import os

def prepareFolders(videoId,folder):
    print(videoId)
    path = os.path.join(folder, videoId)
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def save_and_display_gradcam(img, heatmap, cam_path, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

def denormalize(frame):
    denormalizedFrame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return denormalizedFrame

def get_labels(class_name):
    if class_name == "Avoid":
        return 0
    elif class_name == "Transfer":
        return 1
    else:
        return 2

def load_data(video_pth,class_label,style):
    video_data=[]
    labels=[]
    cap = cv2.VideoCapture(video_pth)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        try:
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            if i % 2 == 0 and style == 'even':
                video_data.append(frame)
            if i % 2 != 0 and style == 'odd':
                video_data.append(frame)
            i += 1
        except:
            break
    cap.release()
    labels.append(get_labels(class_label))

    train_x, x_label = np.array(video_data), np.array(labels)
    train_x = np.expand_dims(train_x, axis=0)
    return train_x, x_label

data_folder="Data/GradCAM"
video_predictor = tf.keras.models.load_model('saved_models/Best_Weights/final_model_loss/')

independent_excel = 'Excel/ClassLabels.xlsx'
sheet_name='Set1_IndependentDataset'
dataDF = pd.read_excel(independent_excel, sheet_name=sheet_name, header=0, index_col=None)

independent_folder = 'Data/Set1_IndependentDataset/*.avi'
video_pths = glob(independent_folder)
styles = ['even','odd']

for video_pth in video_pths:
    video_id = video_pth.split("/")[-1].split('.a')[0]
    row_id = dataDF.index[dataDF['Video_Id'] == video_id]
    class_name = dataDF['Outcome'][row_id].item()
    dataFolder = prepareFolders(video_id, data_folder)
    for style in styles:
        train_x, x_label = load_data(video_pth, class_name, style)

        preds = video_predictor.predict(train_x)
        #print("Prediction is", np.argmax(preds[0]))

        grad_model = keras.models.Model(video_predictor.inputs, [video_predictor.get_layer('residual_main_3').output,
                                                                 video_predictor.output])
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(train_x)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        train_x = train_x[0]
        if style == 'even':
            i=0
        else:
            i=1
        for embryo, hm in zip(train_x, heatmap):
            cam_path = dataFolder+"/cam_" + str(i) + ".jpg"
            embryo = denormalize(embryo)
            save_and_display_gradcam(embryo, hm, cam_path)
            i += 2