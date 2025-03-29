import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
from pypylon import pylon
import time
import math
from itertools import groupby
import json
import os

############################################################### variables ###########################
a=230
b=310
u_all=[]
v_all=[]
tan_theta_all=[]
counter=0
import serial
first_row_laser=20
last_row_laser=1160
noise_detection_threshold=1300
tan_theta_list2=[]
median_value = None
noise_lower_detection_threshold=400
consecutive_found = False
ser=serial.Serial('/dev/ttyS0',9600,timeout=1)
ser.reset_input_buffer()
save_directory = "/home/pi/Desktop/saved_images/Baseline_of_new_method"
save_directory_binary='/home/pi/Desktop/saved_images/Binary_new_method'
noise_upper_detection_threshold=9000
zw0=39                     ### mm
threshold_v_noise=2
threshold_u_noise=2
threshold_tan_noise=2
threshold_for_deltau=8
p=100    ### analogwrite value in Arduino for first laser is 100
k=0     ### jth ub or vb
while True:
    median_array=[]
    v_list=[]
    u_list=[]
    tan_theta_list=[]
    deltau=[]
    deltav=[]
    A_list=[]
    indices=[]
    
    filtered_values_u=[]
    v_filtered=[]
    omitted_indices_u=[]
  

    try:
        
        if ser.in_waiting>0:
             
             read_signal=ser.read()
            
             print(read_signal)
            
             
            
             
            
             if read_signal== b'A': ## equivalent to A
                 p=p+1
                 k=k+1
                 counter=counter+1
                  ## image code
              
                 camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                 camera.Open()
                 
                 print("camera got open")
                 #img_path="testt.jpg"


                 #image=cv2.imread(img_path)
                
                 camera.ExposureTimeAbs=800
                 camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                 
                 grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                 height=grabResult.Height
                 
                 image = cv2.cvtColor(grabResult.Array, cv2.COLOR_BayerBG2RGB)
#                  image_path='/home/pi/Desktop/saved_images/chicken/chicken.jpg'
#                  image=cv2.imread(image_path)
                 
                 ################ delete countours that are too small or large ######################
                 R,G,B =cv2.split(image)
                 ret,green_binary=cv2.threshold(G,7,255,cv2.THRESH_BINARY)
                 ret,Blue_binary=cv2.threshold(B,1,255,cv2.THRESH_BINARY)
                 Blue_binary = cv2.bitwise_not(Blue_binary)
                 ret,Red_binary=cv2.threshold(R,1,255,cv2.THRESH_BINARY)
                 Red_binary = cv2.bitwise_not(Red_binary)
                 bitwise_and_image = cv2.bitwise_and(green_binary, cv2.bitwise_and(Red_binary, Blue_binary))
                 _,countours, _ =cv2.findContours(bitwise_and_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                 for countour in countours:
                       if cv2.contourArea(countour)<noise_detection_threshold:
                          x, y, w, h = cv2.boundingRect(countour)
                          cv2.rectangle(bitwise_and_image, (x, y), (x + w, y + h), (0, 0, 0), thickness=cv2.FILLED)
                       
                 
                # cv2.imshow('Binary',bitwise_and_image)
                 #cv2.waitKey(0)
        
                # kernel = np.ones((3, 3), np.uint8)
                 #opened_image = cv2.morphologyEx(bitwise_and_image, cv2.MORPH_OPEN, kernel)
                 #closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
                 #bitwise_and_image=closed_image 
                 ######################################################################################
                 height, width = bitwise_and_image.shape
                 white_pixels=[]
                 for row_idx in range(first_row_laser, last_row_laser):
                    # Get the row as a 1D array
                    # row_pixels = bitwise_and_image[row, :]
                     row = bitwise_and_image[row_idx, :]
                     white_pixel_indices = np.nonzero(row)[0]
                      
                     u_list_noise=[]
                     filtered_values_u=[]

                   # Check if there are any white pixels in the row
                     
                         
                       # Find the column index 
                     col_index =   white_pixel_indices.tolist()
                    # print("lenght of col_index:",col_index)
                     
                         
                         
                    
                     
                     ### interpolation 
                     if len(col_index)==0 or len(col_index)==1:
                         #current_u_list_idx=(row_idx-first_row_laser)+1
                         current_u_list_idx=(row_idx-first_row_laser)
                         ######## If our first pixel is missing we cannot get data from previous pixel we get the
                         ########## Center pixel coordinates of line and assign it for first pixel
                         if row_idx==first_row_laser:
                            _,contours, _ =cv2.findContours(bitwise_and_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

                            if len(contours) > 0:
                               largest_contour = max(contours, key=cv2.contourArea)
                               M = cv2.moments(largest_contour)
                               if M['m00'] != 0:
                                  centroid_u = int(M['m10'] / M['m00'])
                                  centroid_v = int(M['m01'] / M['m00'])
                                  u_list.append(centroid_u)
                                  v_list.append(row_idx)
        
                  
                         ######### For pixels that are missing but are after first pixel ( sue the previous pixel data)
                         else:
                        
                          prev_u_list_idx=current_u_list_idx-1
                          #prev2_u_list_idx=current_u_list_idx-3
                          #prev3_u_list_idx=current_u_list_idx-4
                         
                          u_avg_prev=u_list[prev_u_list_idx]
                         #u_avg_prev2=u_list[prev2_u_list_idx]
                         #u_avg_prev3=u_list[prev3_u_list_idx]
                          
                          u_avg= u_avg_prev
                          u_avg=int(u_avg)
                          u_list.append(u_avg)
                          v_list.append(row_idx)
                         
                     else:
                      
                         
                      u_list_noise=col_index
                     
                  
                      z_scores_u = np.abs((u_list_noise - np.mean(u_list_noise)) / np.std(u_list_noise))
                     
                      for value, z_score in zip(u_list_noise, z_scores_u) :
                     
                          if z_score <= threshold_v_noise:
                              filtered_values_u.append(value)
                               
                      u_avg=np.average(filtered_values_u)
                      u_avg=int(u_avg)
                      
                     
                      u_list.append(u_avg)
                      
                      v_list.append(row_idx)
                     
                        
                        
                        
                        

                    
                
           
                      
                       
                         
                         
                 
                 u_all.append(u_list)
                 v_all.append(v_list)
                 
                # image_path=os.path.join('/home/pi/Desktop/saved_images/final_results/Images_of_new_method/laser up/raw_image',f'image_{counter}.png')
                 #cv2.imwrite(image_path, image)
                 for i in range(len(v_list)):    
                    cv2.circle(image,(u_list[i],v_list[i]),1,(255,0,00),-1)
                 #cv2.imshow("base_line",image)
                 #cv2.waitKey(0)
                 print("duty is",p)
                
                 
                 
                 
                 
                 

                 
                 ####### Saving Images ########################
#                  
                 image_path = os.path.join('/home/pi/Desktop/saved_images/final_results/Images_of_new_method/Laser_up_second/Baseline_Binary', f'image_{counter}.png')
                 cv2.imwrite(image_path, bitwise_and_image)
#                 
                 image_path=os.path.join('/home/pi/Desktop/saved_images/final_results/Images_of_new_method/Laser_up_second/baseline',f'image_{counter}.png')
                 cv2.imwrite(image_path, image)
                 
                 ######################################
                 
                 
                 
                 if(p==255):
                    # np.savez_compressed("tan_theta_list_duty_4 ",  tan_theta_all)
                     np.savez_compressed('vb_list', v_all)
                     np.savez_compressed('ub_list',  u_all)
                    
                   
                     print("saved")
                     
               
              
                
                
                
                    
                 ser.write(b'C') ### send C (168) so that arduino chage duty cycle
                 
    except OSError as e:
        if e.errno == 25:
            pass 
