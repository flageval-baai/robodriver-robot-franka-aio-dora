import os
import time
import numpy as np
import cv2
import pyarrow as pa
import pyzed.sl as sl
from dora import Node
from enum import Enum


RUNNER_CI = True if os.getenv("CI") == "true" else False


class CaptureMode(Enum):
    LEFT_AND_RIGHT = 1
    LEFT_AND_DEPTH = 2
    LEFT_AND_DEPTH_16 = 3


def encode_frame(frame, encoding):
    if frame is None:
        return None

    if encoding == "bgr8":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif encoding == "rgb8":
        return frame
    elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
        try:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ret, encoded = cv2.imencode("." + encoding, bgr_frame)
            return encoded if ret else None
        except Exception:
            return None
    else:
        return frame


def valid_frame(frame: np.ndarray) -> bool:
    if frame is None:
        return False
    if not isinstance(frame, np.ndarray):
        return False
    if frame.size == 0:
        return False
    if frame.ndim < 2:
        return False
    return True


def safe_resize_and_convert(frame, width, height):
    """
    统一处理：
    - 空帧检查
    - resize
    - BGRA/BGR -> RGB
    """
    if not valid_frame(frame):
        return None

    try:
        frame = cv2.resize(frame, (width, height))
    except Exception:
        return None

    # ZED 通常输出 BGRA
    if frame.ndim == 3:
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame


def main():
    flip = os.getenv("FLIP", "")
    device_serial = os.getenv("DEVICE_SERIAL", "")
    image_height = int(os.getenv("IMAGE_HEIGHT", "480"))
    image_width = int(os.getenv("IMAGE_WIDTH", "640"))
    encoding = os.getenv("ENCODING", "rgb8")
    capture_mode = int(os.getenv("CAPTURE_MODE", "3"))

    app_mode = CaptureMode(capture_mode)

    # ---------------- ZED INIT ----------------
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    if app_mode == CaptureMode.LEFT_AND_RIGHT:
        init_params.depth_mode = sl.DEPTH_MODE.NONE
    else:
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA

    if device_serial:
        init_params.set_from_serial_number(int(device_serial))

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise ConnectionError(f"ZED camera failed to open: {err}")

    camera_info = zed.get_camera_information()

    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_image = sl.Mat()

    # ---------------- DORA NODE ----------------
    node = Node()
    start_time = time.time()

    print(f"[INFO] ZED Camera ready: serial={device_serial}, mode={app_mode.name}")

    for event in node:
        if RUNNER_CI and time.time() - start_time > 10:
            break

        if event["type"] == "INPUT" and event["id"] == "tick":

            if zed.grab() != sl.ERROR_CODE.SUCCESS:
                continue

            # -------- LEFT IMAGE --------
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            left_raw = left_image.get_data()
            left_frame = safe_resize_and_convert(
                np.asanyarray(left_raw) if left_raw is not None else None,
                image_width,
                image_height,
            )

            if left_frame is None:
                continue

            # -------- RIGHT / DEPTH --------
            right_frame = None

            if app_mode == CaptureMode.LEFT_AND_RIGHT:
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                raw = right_image.get_data()
                right_frame = safe_resize_and_convert(
                    np.asanyarray(raw) if raw is not None else None,
                    image_width,
                    image_height,
                )

            elif app_mode == CaptureMode.LEFT_AND_DEPTH:
                zed.retrieve_image(right_image, sl.VIEW.DEPTH)
                raw = right_image.get_data()
                right_frame = safe_resize_and_convert(
                    np.asanyarray(raw) if raw is not None else None,
                    image_width,
                    image_height,
                )

            elif app_mode == CaptureMode.LEFT_AND_DEPTH_16:
                zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
                raw = depth_image.get_data()
                if raw is not None:
                    depth = np.asanyarray(raw)
                    if depth.size != 0:
                        depth = cv2.resize(depth, (image_width, image_height))
                        depth = np.nan_to_num(depth, nan=0.0, posinf=65535, neginf=0.0)
                        right_frame = np.clip(depth, 0, 65535).astype(np.uint16)

            if right_frame is None:
                continue

            # -------- FLIP --------
            if flip == "VERTICAL":
                left_frame = cv2.flip(left_frame, 0)
                right_frame = cv2.flip(right_frame, 0)
            elif flip == "HORIZONTAL":
                left_frame = cv2.flip(left_frame, 1)
                right_frame = cv2.flip(right_frame, 1)
            elif flip == "BOTH":
                left_frame = cv2.flip(left_frame, -1)
                right_frame = cv2.flip(right_frame, -1)

            # -------- CALIB --------
            calib = camera_info.camera_configuration.calibration_parameters
            left_calib = calib.left_cam
            right_calib = calib.right_cam

            # -------- LEFT OUTPUT --------
            meta_l = event["metadata"].copy()
            meta_l.update({
                "encoding": encoding,
                "width": left_frame.shape[1],
                "height": left_frame.shape[0],
                "capture_mode": app_mode.name,
                "principal_point": [int(left_calib.cx), int(left_calib.cy)],
                "focal_length": [int(left_calib.fx), int(left_calib.fy)],
                "timestamp": time.time_ns(),
            })

            left_encoded = encode_frame(left_frame, encoding)
            if left_encoded is None:
                continue

            node.send_output(
                "left_image",
                pa.array(left_encoded.ravel()),
                meta_l,
            )

            # -------- RIGHT OUTPUT --------
            meta_r = event["metadata"].copy()
            meta_r.update({
                "width": right_frame.shape[1],
                "height": right_frame.shape[0],
                "capture_mode": app_mode.name,
                "timestamp": time.time_ns(),
            })

            if app_mode == CaptureMode.LEFT_AND_RIGHT:
                meta_r["encoding"] = encoding
                meta_r["data_type"] = "right_camera"
                encoded = encode_frame(right_frame, encoding)
                output = "right_image"

            elif app_mode == CaptureMode.LEFT_AND_DEPTH:
                meta_r["encoding"] = encoding
                meta_r["data_type"] = "depth_view"
                encoded = encode_frame(right_frame, encoding)
                output = "depth_image"

            else:
                meta_r["encoding"] = "16UC1"
                meta_r["data_type"] = "depth_16bit"
                meta_r["depth_unit"] = "millimeter"
                encoded = right_frame
                output = "depth_data"

            if encoded is None:
                continue

            node.send_output(
                output,
                pa.array(encoded.ravel()),
                meta_r,
            )

        elif event["type"] == "ERROR":
            print("[ERROR]", event["error"])

        elif event["type"] == "STOP":
            break

    zed.close()


if __name__ == "__main__":
    main()
