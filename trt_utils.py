import cv2
import numpy as np
import tensorrt as trt
import scipy.special
import math
import time
import common

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
image_width = 1640
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

MODEL_CONFIG = {
    "tusimple": {
        "input": (800, 288),
        "row_anchor": [
            64,  68,  72,  76,  80,  84,  88,  92,  96,
            100, 104, 108, 112, 116, 120, 124, 128, 132,
            136, 140, 144, 148, 152, 156, 160, 164, 168,
            172, 176, 180, 184, 188, 192, 196, 200, 204,
            208, 212, 216, 220, 224, 228, 232, 236, 240,
            244, 248, 252, 256, 260, 264, 268, 272, 276,
            280, 284
        ],
        "griding_num": 100,
        "num_per_lane": 56,
        "output": (101, 56, 4)
    },
    "culane": {
        "input": (800, 288),
        "row_anchor": [
            121, 131, 141, 150, 160, 170, 180, 189, 199,
            209, 219, 228, 238, 248, 258, 267, 277, 287
        ],
        "griding_num": 200,
        "num_per_lane": 18,
        "output": (201, 18, 4)
    },
}

def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def normalize(im):
    im = np.asarray(im, dtype="float32")
    im = (im / 255.0 - mean) / std
    im = im.transpose(2, 0, 1)
    im = np.expand_dims(im, axis=0)
    return im.astype("float32")

def draw_lane(image, lane_points, color):
    for i in range(len(lane_points) - 1):
        cv2.line(image, tuple(lane_points[i]), tuple(lane_points[i + 1]), color, thickness=2)
def draw_caption(image, box, caption,color):
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1
    )
def calculate_lane_width_in_pixels(left_lane, right_lane):
    # Extract the x-coordinates at the bottom of the image (where y=586 for both lanes)
    left_lane_bottom_x = left_lane[0][0]  # x-coordinate of the left lane at y=586
    right_lane_bottom_x = right_lane[0][0]  # x-coordinate of the right lane at y=586
    
    # Calculate lane width in pixels
    lane_width_pixels = abs(right_lane_bottom_x - left_lane_bottom_x)
    return lane_width_pixels

def estimate_pixel_to_meter_ratio(lane_width_pixels, real_lane_width_meters=3.5):
    # Estimate pixel-to-meter ratio
    return real_lane_width_meters / lane_width_pixels

def calculate_distance_to_left_lane(left_lane_bottom_x, vehicle_center_x, pixel_to_meter_ratio):
    # Calculate the distance from vehicle center to the left lane in pixels
    distance_pixels = abs(vehicle_center_x - left_lane_bottom_x)
    
    # Convert distance from pixels to meters
    distance_meters = distance_pixels * pixel_to_meter_ratio
    return distance_meters
def average_distance_between_points(lane):
    total_distance = 0
    for i in range(len(lane) - 1):
        # Calculate Euclidean distance between consecutive points
        point1 = np.array(lane[i])
        point2 = np.array(lane[i + 1])
        distance = np.linalg.norm(point2 - point1)
        total_distance += distance
    return total_distance / (len(lane) - 1) if len(lane) > 1 else float('inf')
def process_frame(frame, engine, config):
    h, w, _ = frame.shape
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_im = cv2.resize(im, config["input"])
    normalized_im = normalize(resized_im)

    # Inference
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    inputs[0].host = np.ascontiguousarray(normalized_im)
    start_time = time.perf_counter()
    outputs = common.do_inference_v2(
        context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
    )
    end_time = time.perf_counter()
    inference_time_ms = (end_time - start_time) * 1000
    print(f"Inference time: {inference_time_ms:.2f} ms")

    output = outputs[0]
    output = output.reshape(config["output"])
    output = output[:, ::-1, :]
    prob = scipy.special.softmax(output[:-1, :, :], axis=0)
    idx = np.arange(config["griding_num"]) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    output = np.argmax(output, axis=0)
    loc[output == config["griding_num"]] = 0
    output = loc

    col_sample = np.linspace(0, 800 - 1, config["griding_num"])
    col_sample_w = col_sample[1] - col_sample[0]

    lanes_points = []
    lanes_detected = []

    max_lanes = output.shape[1]
    for lane_num in range(max_lanes):
        lane_points = []
        if np.sum(output[:, lane_num] != 0) > 2:
            for point_num in range(output.shape[0]):
                if output[point_num, lane_num] > 0:
                    x = int(output[point_num, lane_num] * col_sample_w * w / config["input"][0]) - 1
                    y = int(h * (config["row_anchor"][config["num_per_lane"] - 1 - point_num] / config["input"][1])) - 1
                    lane_points.append([x, y])
                    
                    
        else:
            lanes_detected.append(False)
        lanes_points.append(lane_points)
    lane_distances = [(average_distance_between_points(lane), lane) for lane in lanes_points if lane]
    sorted_lanes = sorted(lane_distances, key=lambda x: x[0])
# Get the two lanes with the minimum average distances
    lane1 = sorted_lanes[0][1]  # Lane with the smallest average distance
    lane2 = sorted_lanes[1][1]
        
    A_point = lane1[0]
    C_point = lane2[0]
       
    r_l=[]
    l_l=[]
    
    
    if(A_point[0]<C_point[0]):
        r_l=lane2
        l_l=lane1

    else:
        r_l=lane1
        l_l=lane2
    

    left_lane_bottom_x = l_l[0][0]
    right_lane_bottom_x = r_l[0][0]  # This is 68 from the left_lane data

# Calculate lane width in pixels (using previously provided right lane data)

    lane_width_pixels = calculate_lane_width_in_pixels(l_l, r_l)
    pixel_to_meter_ratio = estimate_pixel_to_meter_ratio(lane_width_pixels)
    print(f"Pixel to meter ratio: {pixel_to_meter_ratio:.5f} meters per pixel")

# Calculate the vehicle's center in pixels (assuming vehicle is in the center of the image)
    vehicle_center_x = image_width / 2

# Calculate distance from vehicle center to left lane in meters
    distance_to_left_lane = calculate_distance_to_left_lane(left_lane_bottom_x, vehicle_center_x, pixel_to_meter_ratio)
    distance_to_right_lane = calculate_distance_to_left_lane(right_lane_bottom_x, vehicle_center_x, pixel_to_meter_ratio)
    distance_left_lane= f"Distance to the left lane: {distance_to_left_lane:.2f} meters."
    distance_right_lane=f"Distance to the right lane: {distance_to_right_lane:.2f} meters."

    color=(0,255,0)
    color1=(0,0,255)
    draw_caption(frame,(10,30),distance_left_lane,color=color)
    draw_caption(frame,(10,70),distance_right_lane,color=color1)
    draw_lane(frame, r_l, (0, 255, 0))
    draw_lane(frame, l_l, (0, 0, 255))

    

    return lanes_points













