import cv2
import argparse
from datetime import datetime

vid = cv2.VideoCapture(0) 
contoured_frame_path = ""
import logging 
# video Inference
def save_frame_with_timestamp(frame, output_dir='.', file_prefix='frame'):
    # Open video capture
    # Generate a filename with the current date and time
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'{file_prefix}_{timestamp}.jpg'
    file_path = f'{output_dir}/{filename}'
    # Save the frame to the file
    cv2.imwrite(file_path, frame)
    logging.info(f"Frame saved as {file_path}")

def vid_inf(contoured_path,normal_path,remote_contoured_path,remote_normal_path):
    backSub = cv2.createBackgroundSubtractorMOG2()
    while(True): 
        
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read()
        normal_frame = frame.copy()
        if ret:
            # Apply background subtraction
            fg_mask = backSub.apply(frame)
        # Display the resulting frame
        # cv2.imshow('frame', frame)        
        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        retval, mask_thresh = cv2.threshold( fg_mask, 180, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        min_contour_area = 500  # Define your minimum area threshold
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        frame_out = frame.copy()
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
        if len(large_contours) > 0:
            save_frame_with_timestamp(normal_frame,normal_path)
            save_frame_with_timestamp(frame_out,contoured_path)
        # Display the resulting frame
        cv2.imshow('Frame_final', frame_out)
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 


def main():
      # Create the parser
    parser = argparse.ArgumentParser(description="Process some paths.")

    # Add arguments
    parser.add_argument('--contoured_path', type=str, required=True, help="Path to the contoured file")
    parser.add_argument('--normal_path', type=str, required=True, help="Path to the normal file")
    parser.add_argument('--remote_contoured_path', type=str, required=True, help="Path to the normal file")
    parser.add_argument('--remote_normal_path', type=str, required=True, help="Path to the normal file")

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    contoured_path = args.contoured_path
    normal_path = args.normal_path
    remote_contoured_path = args.remote_contoured_path
    remote_normal_path = args.remote_normal_path
    vid_inf(contoured_path,normal_path,remote_contoured_path,remote_normal_path)

main()