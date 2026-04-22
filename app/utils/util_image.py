import base64
import cv2
import numpy as np

def capture_to_base64(capture_image):
  if isinstance(capture_image, str):
    base64_image = capture_image
  elif hasattr(capture_image, "data"):
    base64_image = capture_image.data
  elif isinstance(capture_image, dict) and "data" in capture_image:
    base64_image = capture_image["data"]
  else:
    base64_image = ""
  return base64_image

def base64_to_cv2(base64_image):
  # 헤더가 포함되어 있다면 (,)를 기준으로 데이터 부분만 추출
  if "," in base64_image:
    base64_image = base64_image.split(",")[1]
  bytes_data = base64.b64decode(base64_image)
  nparr_data = np.frombuffer(bytes_data, dtype=np.uint8)
  cv2_image = cv2.imdecode(nparr_data, cv2.IMREAD_COLOR)
  return cv2_image

def base64_to_jpg(base64_image):
  cv2_image = base64_to_cv2(base64_image)
  _, buffer = cv2.imencode(".jpg", cv2_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
  base64_image = base64.b64encode(buffer).decode("utf-8")
  return base64_image

def base64_resize(base64_image, pixel_size):
  cv2_image = base64_to_cv2(base64_image)
  cv2_image = cv2_resize(cv2_image, pixel_size)
  return cv2_to_base64(cv2_image)

def bytes_to_base64(bytes_image):
  nparr = np.frombuffer(bytes_image, np.uint8)
  cv2_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  _, buffer = cv2.imencode(".jpg", cv2_image)
  base64_image = base64.b64encode(buffer).decode("utf-8")
  return base64_image

def bytes_to_cv2(bytes_image):
  nparr = np.frombuffer(bytes_image, np.uint8)
  cv2_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  return cv2_image

def bytes_resize(bytes_image, pixel_size):
  cv2_image = bytes_to_cv2(bytes_image)
  cv2_image = cv2_resize(cv2_image, pixel_size)
  return cv2_to_bytes(cv2_image)

def cv2_to_base64(cv2_image):
  # cv2 와 numpy 같아요
  _, buffer = cv2.imencode(".jpg", cv2_image)
  base64_image = base64.b64encode(buffer).decode("utf-8")
  return base64_image

def cv2_to_bytes(cv2_image):
  _, buffer = cv2.imencode(".jpg", cv2_image)
  bytes_image = buffer.tobytes()
  return bytes_image

def cv2_resize(cv2_image, pixel_size):
  h, w = cv2_image.shape[:2]
  if w >= h:
    target_w = pixel_size
    target_h = int(h * (pixel_size / w))
  else:
    target_h = pixel_size
    target_w = int(w * (pixel_size / h))
  return cv2.resize(cv2_image, (target_w, target_h), interpolation=cv2.INTER_AREA)

def cv2_letterbox(cv2_image, box=(640, 640), color=(255, 255, 255)):
  if isinstance(box, int):
    box = (box, box)

  h, w = cv2_image.shape[:2]
  target_h, target_w = box

  # 비율을 유지하며 리사이즈
  scale = min(target_w / w, target_h / h)
  new_w, new_h = int(w * scale), int(h * scale)
  resized = cv2.resize(cv2_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

  # 패딩 계산
  pad_w = target_w - new_w
  pad_h = target_h - new_h
  pad_left = pad_w // 2
  pad_right = pad_w - pad_left
  pad_top = pad_h // 2
  pad_bottom = pad_h - pad_top

  # 중앙 정렬된 흰색 배경
  boxed = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
  return boxed