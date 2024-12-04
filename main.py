import argparse
import cv2
from trt_utils import get_engine, process_frame, draw_lane, MODEL_CONFIG
import time
from yoloDet import YoloTRT

WINDOW_NAME = "Test"


    
    
def process_video(input_video_path, output_video_path, yolo_model,yolo_model2, engine, config):
    """
    Processes a video file using YOLOv5 and Ultra-Fast-Lane-Detection models.
    """
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video file: {input_video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video properties: {width}x{height} at {fps} FPS, {frame_count} frames total")
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs if needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        print(f"[INFO] Processing frame {frame_idx + 1}/{frame_count}...")
        start=time.perf_counter()
        
        # Process the frame with the lane detection model
        lanes_points = process_frame(frame, engine, config)
        
        # for lane_points in lanes_points:
        #     if lane_points:
        #         draw_lane(frame, lane_points, (0, 255, 0))  # Green color for lanes
        
        # Process the frame with YOLO model
        yolo_detections, yolo_inference_time = yolo_model.Inference(frame)
        
        yolo_detections2, yolo_inference_time = yolo_model2.Inference(frame)
       
        end=time.perf_counter()

        elapsed_time=end-start


        print(f"Elapsed time:{elapsed_time:.6f} seconds ")
        # Write the processed frame to the output video
        out.write(frame)
                # Display
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        frame_idx += 1
    
    # Release video resources
    cap.release()
    out.release()
    print(f"[INFO] Processed video saved to {output_video_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="File path of TRT model.", required=True)
    parser.add_argument(
        "--model_config",
        type=str,
        default="tusimple",
        help='The name of the model. Either "tusimple" or "culane".',
    )
    parser.add_argument("--videopath", help="File path of input video file.", required=True)
    parser.add_argument("--output", help="File path of output video.", required=True)
    args = parser.parse_args()

    # Load model.
    config = MODEL_CONFIG[args.model_config]
    engine = get_engine(args.model)


    yolo_model = YoloTRT(
        library="/home/shirel/JetsonYolov5/yolov5/build/libmyplugins.so",
        engine="/home/shirel/JetsonYolov5/yolov5/build/yolov5s.engine",
        conf=0.5,
        yolo_ver="v5",
        categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"])

    yolo_model2 = YoloTRT(
        library="/home/shirel/JetsonYolov5/yolov5/build/libmyplugins.so",
        engine="/home/shirel/APP/JetsonYolov5/yolov5/build/s.engine",
        conf=0.3,
        yolo_ver="v5",
        categories = ["License Plate"]
    )
    
    # Process the video
    process_video(args.videopath, args.output, yolo_model,yolo_model2, engine, config)
    
    
    



    del engine

if __name__ == "__main__":
    main()