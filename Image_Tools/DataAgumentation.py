import cv2
import numpy as np
import os 

'''Shifts the HSV colors of images in a folder and saves them safely.  
Args:
    src_path (str): Path to the folder containing source images.
    hue_shift (int/float): Amount to rotate the color wheel (0-179 in OpenCV).
    saturation_scale (float): Multiplier for color intensity/vibrancy.
    value_scale (float): Multiplier for image brightness.
Return :
        None'''
    
def color_shift(src_path,hue_shift, saturation_scale, value_scale):
    # Define and create a unique destination folder to prevent overwriting original files
    dst_path=os.path.join(src_path,"ColorShift")
    os.makedirs(dst_path,exist_ok=True)
    files=os.listdir(src_path)
    for file in files:
        img_path = os.path.join(src_path,file)
        # Skip folders
        if os.path.isdir(img_path):
            continue
        image=cv2.imread(img_path)
        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Split the HSV image into individual channels
        h, s, v = cv2.split(hsv_image)

        # Apply color shifts to the individual channels
        h_shifted = cv2.add(h, hue_shift)
        s_scaled = cv2.multiply(s, saturation_scale)
        v_scaled = cv2.multiply(v, value_scale)

        # Clip the values to the valid range of each channel
        h_shifted = np.clip(h_shifted, 0, 179)
        s_scaled = np.clip(s_scaled, 0, 255)
        v_scaled = np.clip(v_scaled, 0, 255)

        # Merge the modified channels back into an HSV image
        modified_hsv = cv2.merge((h_shifted, s_scaled, v_scaled))

        # Convert the modified HSV image back to the BGR color space
        modified_bgr = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(dst_path+"\\"+file,modified_bgr)
        print(f"Image Saved")

'''Adjust Brightness of images in a folder and saves them safely.  
Args:
    src_path (str): Path to the folder containing source images.
    value (int): Multiplier for image brightness.
Return :
        None'''  
def adjust_brightness(src_path,value):
    dst_path=os.path.join(src_path,"BrightNess")
    os.makedirs(dst_path,exist_ok=True)
    files=os.listdir(src_path)
    for file in files:
        img_path = os.path.join(src_path,file)
        # Skip folders
        if os.path.isdir(img_path):
            continue
        image=cv2.imread(img_path)
        # Convert the image to the LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab_image)

        # Apply the brightness adjustment to the L channel
        l_adjusted = cv2.add(l, value)

        # Clip the values to the valid range of the L channel (0-255)
        l_adjusted = np.clip(l_adjusted, 0, 255)

        # Merge the adjusted L channel with the original A and B channels
        adjusted_lab = cv2.merge((l_adjusted, a, b))

        # Convert the adjusted LAB image back to the BGR color space
        adjusted_bgr = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(dst_path+"\\"+file,adjusted_bgr)
        print(f"Image Saved")

        
'''Adjust Contrast of images in a folder and saves them safely.  
Args:
    src_path (str): Path to the folder containing source images.
    value (int): Multiplier for image brightness.
Return :
        None''' 
def adjust_contrast(src_path,value):
    dst_path=os.path.join(src_path,"Contrast")
    os.makedirs(dst_path,exist_ok=True)
    files=os.listdir(src_path)
    for file in files:
        img_path = os.path.join(src_path,file)
        # Skip folders
        if os.path.isdir(img_path):
            continue
        image=cv2.imread(img_path)
        # Convert the image to the LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab_image)

        # Apply the contrast adjustment to the L channel
        l_adjusted = cv2.multiply(l, value)

        # Clip the values to the valid range of the L channel (0-255)
        l_adjusted = np.clip(l_adjusted, 0, 255)

        # Merge the adjusted L channel with the original A and B channels
        adjusted_lab = cv2.merge((l_adjusted, a, b))

        # Convert the adjusted LAB image back to the BGR color space
        adjusted_bgr = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(dst_path+"\\"+file,adjusted_bgr)
        print(f"Image Saved")


# Define the contrast adjustment value
contrast_value = 1.0  # Increase contrast by 1.5 (use values less than 1 to reduce contrast)




if __name__=="__main__":
    #color_shift("G:\\Vian\\240428002",20,0.7,0.8)
    #adjust_brightness("G:\\Vian\\240428002",-50)
    adjust_contrast("G:\\Vian\\240428002",1.5) #Increase contrast by 1.5 (use values less than 1 to reduce contrast)