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
import math
############################################################### variables ###########################
a=560
b=630
du_list=[]
uo_all=[]
vo_all=[]
tan_theta_all=[]
zw_all=[]
std_check=[]
counter=1
import serial
#first_row_laser=350
#last_row_laser=860
first_row_laser=20
last_row_laser=1160
noise_detection_threshold=300
tan_theta_list2=[]
median_value = None
noise_lower_detection_threshold=400
consecutive_found = False
ser=serial.Serial('/dev/ttyS0',9600,timeout=1)
ser.reset_input_buffer()

noise_upper_detection_threshold=9000
zw0=39                     ### mm
threshold_v_noise=5
threshold_u_noise=2
threshold_tan_noise=2
threshold_for_deltau=8
p=100   ### analogwrite value in Arduino
k=0     ### jth ub or vb
##################################################### Functions #############################
def extract_far_from_average(input_list, threshold):
    average = (input_list[1]+ input_list[2]+ input_list[3]+ input_list[4]+ input_list[5]+ input_list[6])/6
    result = []
    for item in input_list:
        if abs(item - average) > threshold:
            result.append(item)
    return result
############################### While Loop ##################
Z=np.zeros((1200, 1920))



while True:
    median_array=[]
    vo_list=[]
    uo_list=[]
    tan_theta_list=[]
    deltau=[]
    deltav=[]
    A_list=[]
    indices=[]
    zw_list=[]
   
    
    filtered_values_u=[]
    v_filtered=[]
    omitted_indices_u=[]
  

    try:
      
       # print("ser.in_waiting",ser.in_waiting)
        if ser.in_waiting>0 :
             
             read_signal=ser.read()
                         
            
             print(read_signal)            
        
             if read_signal== b'A': ## equivalent to A
                 vb_list=np.load('vb2.npy')
                 ub_list=np.load('ub2.npy')
             
                 ub=ub_list[k]
                 vb=vb_list[k]
            
                  ## image code
                 p=p+1
                 k=k+1
             #### calculate tan theta
                 #theta_degrees=-0.1416*(p)+90.711  ## prev theta for laser down
                 theta_degrees=-0.1322*(p)+92.246
                 print("Theta",   theta_degrees)
                     
                 theta_radians = theta_degrees * (0.01744)
               

                 tan = math.tan(theta_radians)
                 
                 
                 camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
                 camera.Open()
                 print("camera got open")
             
                
                 camera.ExposureTimeAbs=1300 # 1300 worked at night
                 camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                 grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                 height=grabResult.Height
                 
                 image = cv2.cvtColor(grabResult.Array, cv2.COLOR_BayerBG2RGB)
                
                 ################ delete countours that are too small or large ######################
                 R,G,B =cv2.split(image)
                 ret,green_binary=cv2.threshold(G,5,255,cv2.THRESH_BINARY)  ### 6 was first try
                 ret,Blue_binary=cv2.threshold(B,1,255,cv2.THRESH_BINARY)
                 Blue_binary = cv2.bitwise_not(Blue_binary)
                 ret,Red_binary=cv2.threshold(R,1,255,cv2.THRESH_BINARY)
                 Red_binary = cv2.bitwise_not(Red_binary)
                 bitwise_and_image = cv2.bitwise_and(green_binary, cv2.bitwise_and(Red_binary, Blue_binary))
                 #kernel = np.ones((5, 5), np.uint8)
                 #opened_image = cv2.morphologyEx(bitwise_and_image, cv2.MORPH_OPEN, kernel)
                 #closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
                 #bitwise_and_image=closed_image
                 _,countours, _ =cv2.findContours(bitwise_and_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                 for countour in countours:
                        if cv2.contourArea(countour)<noise_detection_threshold:
                           x, y, w, h = cv2.boundingRect(countour)
                           cv2.rectangle(bitwise_and_image, (x, y), (x + w, y + h), (0, 0, 0), thickness=cv2.FILLED)
                       
                 
                 #cv2.imshow('Binary',bitwise_and_image)
                 #cv2.waitKey(0)
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
           
                         
                     u_list_noise=col_index
                     
                  
                     z_scores_u = np.abs((u_list_noise - np.mean(u_list_noise)) / np.std(u_list_noise))
                     mean_scores_u=np.abs((u_list_noise - np.mean(u_list_noise)))
                     for value, m_score in zip(u_list_noise, mean_scores_u) :
                     
                         if m_score <= threshold_v_noise:
                             filtered_values_u.append(value)
                     import math
                     filtered_values_u = [value for value in filtered_values_u if not math.isnan(value)]
                     if len(filtered_values_u )>0:
                      uo_avg=np.average(filtered_values_u)
                      
                     
                     
                      uo_avg=int(uo_avg)
                   
                      
                     
                      uo_list.append(uo_avg)
                     
                      vo_list.append(row_idx)
                     
                      
                                       
                         
                 
                 uo_all.append(uo_list)
                 vo_all.append(vo_list)
                 
               
             
              
                 print("duty is",p)
                
                
                 ################ Baseline correction
                 baseline_check_list=[]
                 for i in range(20):
                     d=uo_list[i]-ub[i]
                     baseline_check_list.append(d)
                     
                 average_of_baseline_check_list=np.average( baseline_check_list)
                 print(" average_of_baseline_check_list", average_of_baseline_check_list)
                 if average_of_baseline_check_list>2 :
                     for i in range(len(ub)):
                         ub[i]=ub[i]+int(average_of_baseline_check_list)
                         
                 if average_of_baseline_check_list<-2 :
                     for i in range(len(ub)):
                         ub[i]=ub[i]+int(average_of_baseline_check_list)
                     
                 
                 
                 
               
                
                 
                 
                 
                 
                 
                 
################################################################# FIND Z ########################################################
                 
                 
               
          
                 for uo,u in zip(uo_list,ub):
                     delta=u-uo
                         
                     deltau.append(delta)
                             
                 for i in range(len(uo_list)):
                     A=(-0.00045*uo_list[i]+0.37+0.0000116*vo_list[i])
                     
                     A_list.append(A)
                 for i in range (len(deltau)):
                     
                    #tan_theta=(zw0)/(A_list[i]*zw0+0.0088*deltav[i]-0.639*deltau[i])
                     zw=0.59*deltau[i]*tan/(-1+A_list[i]*tan)
                     #zw=-zw
                  
                     zw_list.append(zw)
               
                 zw_all.append(zw_list)
                 
               
                     
                 for i in range (len(uo_list)):
                   

                        
                     Z[vo_list[i],uo_list[i]]=zw_list[i]
                 
                
         
                
                 ##### MARK baseline blue and object line green
                 
                 for i in range(len(uo_list)):    
                    cv2.circle(image,(uo_list[i],vo_list[i]),1,(255,0,0),-1)       ### Object Pixels
                 for i in range(len(vb)):    
                    cv2.circle(image,(ub[i],vb[i]),1,(0,0,255),-1)                    ## Baseline
                 ####### Baseline VS Object pixels saving Image ########################
                 
                 image_path = os.path.join('/home/pi/Desktop/saved_images/final_results/Images_of_new_method/Laser_up_second/object_and_baseline', f'image_{counter}.png')
                 cv2.imwrite(image_path, image)
                 
                 #name="img"
                 #cv2.imshow(name, image)
                 #cv2.waitKey(0)
                 counter=counter+1
                
                
                 
                 ###################################### Saving Binary Image of Object Pixels#######################
                # image_path = os.path.join('/home/pi/Desktop/saved_images/final_results/Images_of_new_method/laser up/obj_binary', f'image_{counter}.png')
                 #cv2.imwrite(image_path, bitwise_and_image)
#                  counter=counter+2
                  

                 
                 if(p==255):
                     np.savez_compressed("2 ",  Z)
                     #np.savez_compressed("std_list ", std_check )
                    
                     print("saved")
                    
                     
               
              
                
                
                  
                 
                 ser.write(b'C') ### send C (168) so that arduino chage duty cycle
                 
    except OSError as e:
        print("ERRO")
        print(" U ARE PROBABLY LOADING SOMETHING TO YOUR TRY CODE THAT IS NOT AVAILABE IN THIS DIRECTORY")
        if e.errno == 25:
            pass 
            
