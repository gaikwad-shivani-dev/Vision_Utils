import cv2
import json
import numpy as np
import os
from shapely.geometry import Polygon

def save_frames_from_rtsp(rtsp_url, num_frames_to_save, output_directory,camera1_Kerb_points):
    Flag=1
    frame_count = 0
    # Open the RTSP stream
    if Flag:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print("Error: Unable to open RTSP stream.")
            return

        

        while frame_count < num_frames_to_save:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame.")
                break

            # Save the frame
            frame_filename = f"{output_directory}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {frame_count}")
            # Draw polygons on the image in red color
            for points in camera1_Kerb_points:
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        
            frame_filename = f"{output_directory}/frame_AEntry{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        # Release the capture
        cap.release()
    else:
        frame=cv2.imread("Output\\PARALLELPARKING_camera2_Transformed.jpg")
        frame_filename = f"{output_directory}/frame_{frame_count}_Top.jpg"
        cv2.imwrite(frame_filename, frame)
        for points in camera1_Kerb_points:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        frame_filename = f"{output_directory}/frame_A{frame_count}_Top.jpg"
        cv2.imwrite(frame_filename, frame)
        

def HoleInPoly(rtsp_url, num_frames_to_save, output_directory,camera1_Kerb_points,camera2_Kerb_points):
    frame_count=0
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream.")
        return

    while frame_count < num_frames_to_save:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break
        # Save the frame
        frame_filename = f"{output_directory}/frame_{frame_count}.jpg"
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame {frame_count}")
        exterior = np.array(camera1_Kerb_points, np.int32).reshape((-1, 1, 2))
        interior = np.array(camera2_Kerb_points, np.int32).reshape((-1, 1, 2))

        # Visualize on the frame
        
        cv2.polylines(frame, [exterior], isClosed=True, color=(0, 255, 0), thickness=2) 
        cv2.polylines(frame, [interior], isClosed=True, color=(255, 0, 0), thickness=2) 
        frame_filename = f"{output_directory}/frame_{frame_count}_A.jpg"
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame {frame_count}")
        frame_count += 1


    
if __name__ == "__main__":
    print(os.getcwd())
    rtsp_url = "Input\\LMV\\4\\Cam2_113956626.avi"  # Replace this with your RTSP stream URL
    num_frames_to_save = 1  # Number of frames to save
    output_directory = 'Output\\'  # Directory to save frames
    # Load JSON data from the 'kerb_camera_mapping.json' file
    with open(r"Input\\ViAn_Parked_config.json", 'r') as f:
        json_data = json.load(f)

    # Extract polygon points for Camera1["RTSP_streams"][ComputeMulH.TrackName][CamNo]["rtsp"]
    OuterP = json_data['RTSP_streams']['PARALLELPARKING'][1]['Outercordinates']
    InnerP = json_data['RTSP_streams']['PARALLELPARKING'][1]['Innercordinates']
    Coords = json_data['RTSP_streams']['PARALLELPARKING'][1]['TopCordinates']
    save_frames_from_rtsp(rtsp_url, num_frames_to_save, output_directory,Coords)
    #HoleInPoly(rtsp_url, num_frames_to_save, output_directory,OuterP,InnerP)
