# importing required libraries

import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import numpy as np
from scipy import signal
import cv2

np.seterr(over='ignore')


# Create Average Kernal with specified  size
def Create_Gaussian_filterKernal( Kernal_size,std ):
                gkern1d = signal.gaussian(Kernal_size, std=std).reshape(Kernal_size, 1)
                Gaussian_Kernal = np.outer(gkern1d, gkern1d)
                return Gaussian_Kernal


def Convolution(image,kernal):
            pixel_value=0
            image_array = np.array(image)
            rows, columns = image_array.shape
            Krows,kcolumns=kernal.shape
            new_image=np.ones([rows-(Krows-1), columns-(kcolumns-1)],dtype=np.float32)
    # For Moving Window
            for i in range(rows-( Krows-1)):
                for j in range(columns-(kcolumns-1)):
                   V_index=0
                   U_index=0
# For convolving with kernals depening on their selected size
                   for rp in range (i,i+ Krows):
                        for cp in range (j,j+kcolumns):
                            pixel_value=pixel_value+image_array[rp][cp]*kernal[U_index][V_index]
                            V_index+=1
                        V_index=0
                        U_index+=1
                   U_index=0
                   new_image[i][j]=pixel_value
                   pixel_value=0
            print(new_image)
            return new_image

def normalization(image):
            norm = (image - np.min(image)) / (np.max(image) - np.min(image))*255
            return norm

# Create Average Kernal with specified  size
def Create_box_filterKernal( w ):
    return np.ones((w,w),np.float32)  / (w*w)

# Get Median Value of an array
def get_median(array):
    for i in range(len(array)):
        key=array[i]
        j=i-1 
#Sorting An array by Move elements of arr[0..i-1], that are greater than key, to one position aheadof their current position 
        while(j>=0 and (array[j]>key)):
            array[j+1]=array[j]
            j=j-1
        array[j + 1] = key
# Return the median value after sorting 
    return array[int((len(array)/2)+.5)-1]

# Applying Average Filter 
def Average_Filter(image,Kernal_size):
        kernal=Create_box_filterKernal(Kernal_size)
        new_image=Convolution(image,kernal)
        norm = normalization(new_image)
        return (Image.fromarray(norm))

# Applying Gaussian Filter 
def Gaussian_Filter(image,Kernal_size,std):
        kernal=Create_Gaussian_filterKernal(Kernal_size,std)
        new_image=Convolution(image,kernal)
        print(new_image)
        norm =normalization(new_image)
        return (Image.fromarray(norm))

# Applying Median Filter 
def Median_Filter(image,Kernal_size):
        window_data=[]
        image_arr=np.array(image)
        rows, columns = image_arr.shape
        new_image=np.ones([rows-(Kernal_size-1), columns-(Kernal_size-1)], dtype = int)
# For Moving Window
        for i in range(rows-(Kernal_size-1)):
            for j in range(columns-(Kernal_size-1)):
                # For convolving with kernals depening on their selected size
                for rp in range (i,i+Kernal_size):
                      for cp in range (j,j+Kernal_size):
                            window_data.append(image_arr[rp][cp])
                new_image[i][j]=get_median(window_data) 
                window_data=[]   
        return (Image.fromarray(new_image))


fig = plt.figure()
# opening image using pil and converting image to GrayScale 
Noisy_image = Image.open("saltandpapper.jpg").convert('L')
# Adds a Noisy Img at the 1st position
fig.add_subplot(3,3,1)
plt.imshow(Noisy_image,cmap='gray')
plt.axis('off')
plt.title("Noisy_image")


# Filter Image
Filterd_image=Median_Filter(Noisy_image,3)
fig.add_subplot(3,3,2)
plt.imshow(Filterd_image,cmap='gray')
plt.axis('off')
plt.title("Median Filter")

# Compare to openCV Filtering 
img_cv = cv2.imread('saltandpapper.jpg')
median = cv2.medianBlur(img_cv,3)
fig.add_subplot(3,3,3)
plt.imshow(median,cmap='gray')
plt.axis('off')
plt.title("Median Filter_CV")

Noisy_image2 = Image.open("mona.png").convert('L')
fig.add_subplot(3,3,4)
plt.imshow(Noisy_image2 ,cmap='gray')
plt.axis('off')
plt.title("Noisy_image2")

AVG_Filterd_image=Average_Filter(Noisy_image2,3)
fig.add_subplot(3,3,5)
plt.imshow(AVG_Filterd_image,cmap='gray')
plt.axis('off')
plt.title("Average Filter 3x3")


img_cv2 = cv2.imread('mona.png')
img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
AVG_blur = cv2.blur(img_cv2 ,(3,3))
fig.add_subplot(3,3,6)
plt.imshow(img_gray)
plt.axis('off')
plt.title("open_CV Average Filter 3x3")


fig2 = plt.figure()

Noisy_image3= Image.open("GaussianNoise.jpg").convert('L')
img_cv3 = cv2.imread('GaussianNoise.jpg')
fig2.add_subplot(2,3,1)
plt.imshow(Noisy_image3 ,cmap='gray')
plt.axis('off')
plt.title("Gaussian ")

Gaussian_Filterd_image=Gaussian_Filter(Noisy_image3,5,2)
fig2.add_subplot(2,3,2)
plt.imshow(Gaussian_Filterd_image,cmap='gray')
plt.axis('off')
plt.title("Gaussian  Filter 5x5 sigma=0")


img_cv3 = cv2.imread('GaussianNoise.jpg')
Gaussian_blur = cv2.GaussianBlur(img_cv3 ,(5,5),2)
fig2.add_subplot(2,3,3)
plt.imshow(Gaussian_blur)
plt.axis('off')
plt.title("openCV_Gaussian Filter 5x5 sigma=2")


fig2.add_subplot(2,3,4)
plt.imshow(img_cv3 ,cmap='gray')
plt.axis('off')
plt.title("Gaussian ")

Gaussian_Filterd_image=Gaussian_Filter(Noisy_image3,5,2)
fig2.add_subplot(2,3,5)
plt.imshow(Gaussian_Filterd_image,cmap='gray')
plt.axis('off')
plt.title("Gaussian  Filter 5x5 sigma=1")

Gaussian_blur = cv2.GaussianBlur(img_cv3 ,(5,5),8)
fig2.add_subplot(2,3,6)
plt.imshow(Gaussian_blur)
plt.axis('off')
plt.title("openCV_Gaussian Filter 5x5 sigma=1")
plt.show()
