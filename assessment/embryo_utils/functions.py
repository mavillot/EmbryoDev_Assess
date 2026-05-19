import cv2
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from PIL import Image
import statistics
import sys
from scipy.ndimage import generic_filter
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.filters.rank.generic import threshold
from skimage import filters 
from skimage.filters.rank import entropy
from skimage.morphology import disk

def diameter(mask):
    ymin=0
    for row in mask:
        if sum(row)>0:
            break
        ymin+=1
    ymax=len(mask)
    for row in mask[::-1]:
        if sum(row)>0:
            break
        ymax-=1
    return ymax-ymin

def center(mask):
    ymin=0
    for row in mask:
        if sum(row)>0:
            break
        ymin+=1
    ymax=len(mask)
    for row in mask[::-1]:
        if sum(row)>0:
            break
        ymax-=1
    return (ymin+ymax)//2


def hull(mask):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_image = np.zeros_like(mask)
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        cv2.drawContours(hull_image, [hull], 0, 255, -1)
    kernel = np.ones((5, 5), np.uint8)  # A 5x5 square kernel
    hull_image = cv2.erode(hull_image, kernel, iterations=1)
    hull_image = cv2.dilate(hull_image, kernel, iterations=1)
    return hull_image

def intersection(te, icm, N=40):
    # return the exact point of the intersection between TE and ICM after the max is obtained
    te=np.convolve(te, np.ones(N)/N, mode='same')
    icm=np.convolve(icm, np.ones(N)/N, mode='same')
    intersections=[]
    maxi=list(icm).index(max(icm))
    for i in range (1,len(te)):
        if te[i-1]<icm[i-1] and te[i]>icm[i] and i>maxi:
            intersections.append(i)
    if intersections!=[]:
        pos=intersections[0]
    else:
        pos=len(te) - 2
    return pos     



def polar_img(x_band, y_band):
    if len(x_band) == 0 or len(y_band) == 0:
        return None
    polar_coords = np.zeros((round(np.max(y_band))+1,360))
    for i in range(360):  # Fix: Use range(360) instead of range(0, 361)
        indices = np.where((x_band >= i) & (x_band < i + 1))  # Find indices close to `i`
        if len(indices[0]) > 0:  # Ensure we found matching points
            r_values = y_band[indices].astype(int)  # Convert to int for valid indexing
            polar_coords[r_values, i] = 255  # Assign value
    return polar_coords

def radius(polar_coords):
    R, r = [],[]
    for x in range(0,360):
        try:
            R.append(max(np.where(polar_coords[:,x]==255)[0]))
        except:
            pass
        try:
            r.append(min(np.where(polar_coords[:,x]==255)[0]))
        except:
            pass
    return np.mean(R), np.mean(r)

def thikness(polar_band):
    thickness = []
    for x in range(0,360):
        idx = np.where(polar_band[:,x]==255)[0]
        if len(idx)>0:
            thickness.append(max(idx)-min(idx))
        else :
            thickness.append(0)
    return thickness

# Function to compute the fractal dimension using the box-counting method
def higuchi_fractal_dimension(time_series, k_max):
    N = len(time_series)
    L = []
    k_values = range(1, k_max + 1)
    
    for k in k_values:
        Lk = []
        for m in range(k):
            X_mk = time_series[m:N:k]  # Extract subsequence
            n_k = len(X_mk)
            
            if n_k > 1:
                norm_factor = (N - 1) / (n_k * k)
                length = sum(abs(np.diff(X_mk))) * norm_factor
                Lk.append(length)
        
        L.append(np.mean(Lk))
    
    # Fit a line in log-log space
    log_k = np.log(1 / np.array(k_values))
    log_L = np.log(L)
    coeffs = np.polyfit(log_k, log_L, 1)
    fractal_dimension = abs(coeffs[0])
    
    return fractal_dimension

# Functions for Embryo_video

def is_low(num, mini, maxi):
    return ((num-mini)/(maxi-mini)) <0.1


def n_window(list_values,N):
    windw=[]
    for i in range(1,len(list_values)):
        if list_values[i-1]!=list_values[i]:
            windw.append(i+N)
    return windw

def window(list_values,N):
    mean_smooth=np.convolve(list_values, np.ones(N)/N, mode='same')[N:-N]
    mins=is_low(mean_smooth, min(mean_smooth), max(mean_smooth))
    return n_window(mins,N)


def cnt_elipse(ellipse,yx):
    blank_image = np.zeros((yx[0], yx[1]), dtype=np.uint8)
    cv2.ellipse(blank_image, ellipse, 255, 2)
    cnt=cv2.findContours(blank_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if len(cnt)>0:
        return cnt[0]
    else: return np.array([0])


def intensity_profile(img):
    img=no_back(img)
    N=50
    y,x=img.shape
    mean_row=[np.mean(fila) for fila in img]
    mean_col=[np.mean(img[:,i]) for i in range(0,x)]
    Y=window(mean_row,N)
    X=window(mean_col,N)
    if len(X)>1 and len(Y)>1:
        return (X[0],X[1],Y[0],Y[1])
    else:
        return (0,0,0,0)

def watershed_seg(img): #mask dtype bool
    thresh_nucleo=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
    dist = ndi.distance_transform_edt(thresh_nucleo)
    dist_visual = dist.copy()
    local_max=peak_local_max(dist, min_distance=1, labels=thresh_nucleo)
    mask_ = np.zeros(dist.shape, dtype=bool)
    mask_[tuple(local_max.T)] = True
    markers, _ = ndi.label(mask_)
    markers = cv2.watershed(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB),markers)
    return markers


def no_back(img):
    markers=watershed_seg(img)
    for i in np.unique(markers[[1,-2],:]):
        if np.mean(img[markers==i])<np.mean(img.flatten())/2 and i!=-1:
            img[markers==i]=np.mean(img.flatten())
    return img


def draw_pn(pn):
    n=0
    _,th = cv2.threshold(pn,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    erode=cv2.erode(th,np.ones((3,3)))
    contours, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour)>5:
            convex_hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(convex_hull)
            if hull_area>100:
                M = cv2.moments(convex_hull)
                    # Calculate the centroid coordinates
                centroid_x = int(M['m10'] / M['m00'])
                centroid_y = int(M['m01'] / M['m00'])
                ellipse=cv2.fitEllipse(contour)
                x,y=ellipse[0]
                ellipse = cnt_elipse(ellipse, erode.shape)
                if ellipse.all()!=0:
                    area = cv2.contourArea(ellipse)
                    solidity = area / hull_area
                    if solidity>0.9 and abs(centroid_x - x) <5 and abs(centroid_y - y) <5:
                        n+=1
    return n

def counting_pronuclei(cropped_frame):
    mask=np.zeros_like(cropped_frame)
    X_0,X_1,Y_0,Y_1=intensity_profile(cropped_frame)
    if X_1> 0:
        pn=cropped_frame[Y_0:Y_1,X_0:X_1]
        pn=255*(1-draw_pn(pn))
        pn=cropped_frame[Y_0:Y_1,X_0:X_1]
        pn=255*(1-draw_pn(pn))
        mask[Y_0:Y_1,X_0:X_1]=pn
    num_labels, _ = cv2.connectedComponents(mask)
    return num_labels -1

def n_cells_cleaning(n_cells):
    x = 0
    for i in range(0,len(n_cells)):
        if n_cells[i] is None:
            n_cells[i] = x
        elif n_cells[i] >= x:
            x = n_cells[i]
        else:
            n_cells[i] = x
    return n_cells
