# This is a program to identify the max occupied color and its name from the given image.

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
import os
import webcolors



# Function to convert the RGB values into Hexa-Decimal values
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

# Function to read the Image in OpenCV and convert its color from BGR to RGB
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #plt.imshow("get_image",image)
    #plt.imshow(image)
    #plt.show()
    return image


# Function to find the closest color name , if the exact color name is not defined by CSS .
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

# Function to find the Exact Color name from the Webcolors Package defined by HTML and CSS . 
def get_colour_name(requested_colour):
    requested_colour = [int(s) for s in requested_colour]
    #print("$$$$$$$$$",requested_colour)
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return  closest_name

# Function to convert the centered HSV cluster into RGB values
def convert_HSV2RGB(center_colors,rgbarr=[]):
    #print("Inside HSV2RGB - center colors =======",center_colors)
    #print("shape = ",center_colors.shape)
    for cluster in center_colors:
        #print("cluster %%%%%%%%%%",cluster)
        h = cluster[0]
        s = cluster[1]
        v = cluster[2]
        #print("H,S,V ========",h,s,v)
        # hsv2rgb = colorsys.hsv_to_rgb(h, s, v) 
        hsv_image = np.uint8([[[h,s,v ]]]) 
        #print("hsv_image ***************",hsv_image)
        hsv2rgb = cv2.cvtColor(hsv_image,cv2.COLOR_HSV2RGB)

        h1 = hsv2rgb[0][0][0]
        s1 = hsv2rgb[0][0][1]
        v1 = hsv2rgb[0][0][2]
        #print("h1,s1,v1&&&&&&&&&&&&&&&&&&&&&&&&&",h1,s1,v1)
        rgb_row = [h1,s1,v1]
        #print("RGBrow =============",rgbrow)
        rgbarr.append(rgb_row)
        print("hsv2rgb arr *******************",rgbarr)
    
    return rgbarr

# Function to find the Most Occupied color using K-Means Algorithm .
def get_colors(image, number_of_colors, show_chart):
    #cv2.imshow("Actual image ",image)
    # Resize and Reshape the image
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    # apply K-means algorithm - pass the no. of max. colors to find as argument .
    clf = KMeans(n_clusters = number_of_colors)
    # Fit and predict the labels for the mentioned no. of clusters. Compute cluster centers and predict cluster index for each sample.
    labels = clf.fit_predict(modified_image)
    
    #print("Labels =====",labels)
    # Use Counter to store the no. of pixels available in each cluster
    # A Counter is a subclass of dict. Therefore it is an unordered collection where elements and their respective count are stored as dictionary. 
    counts = Counter(labels)
    #print("counts = = = ",counts)
    
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    #  finding the center color value for each of the clusters
    center_colors = clf.cluster_centers_ 
    #print("centerd_colors =========",center_colors)
    
    #convert the cluster values from HSV2RGB 
    center_colors = convert_HSV2RGB(center_colors,[])
    #print("centerd_colors =========",center_colors)
    
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    
    #hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()] # This returns the Hexa-Decimal value of each cluster
    hex_colors = [get_colour_name(ordered_colors[i]) for i in counts.keys()] # This returns the color name as value of each cluster
    
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    #print(" rgb_colors : ", rgb_colors)      
    
    # TO Display the highly occupied color name 
    global max_occupied_color
    for i,val in Counter(labels).most_common(1):
        max_color = i
        #print("i = ",max_color)
        #print("val = ",val)
        color_name = get_colour_name(ordered_colors[max_color])
        print("Exact color name =====",color_name)
        max_occupied_color = color_name

    # To Display each cluster color in pie chart with percentage
    if(show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors,autopct='%1.0f%%')
        plt.show()
    
    return rgb_colors

# Function to label the identified color name on image
def get_color_label(image):
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    org = (50, 50) 
    # fontScale 
    fontScale = 1
   
    # Blue color in BGR 
    color = (255, 0, 0) 
  
    # Line thickness of 2 px 
    thickness = 2
    
    # Using cv2.putText() method 
    image = cv2.putText(cv2.cvtColor(image, cv2.COLOR_HSV2RGB),max_occupied_color, org, font,fontScale, color, thickness, cv2.LINE_AA) 
    plt.imshow(image)
    plt.show()


IMAGE_DIRECTORY = 'images'
#dir_path = os.getcwd()
dir_path = os.path.dirname(os.path.realpath(__file__))
#print("dir_path =========",dir_path)
Image_Folder_Path =os.path.join(dir_path,IMAGE_DIRECTORY)
#print(Image_Folder_Path)

for file in os.listdir(Image_Folder_Path):
    
    if not file.startswith('.'):
        print("PAth = ",os.path.join(Image_Folder_Path, file))
        Image_Path = os.path.join(Image_Folder_Path, file)
        image1 = get_image(Image_Path)
        get_colors(image1,5,True)
        get_color_label(image1)