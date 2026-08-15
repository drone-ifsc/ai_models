# ai_models

Keypoint models that locate a **square landing base** in a frame from a downward-facing
drone camera: **1 center + 4 corners**. Weights only — no training or inference code;
everything needed to run them is in the examples below.

They were built for vision-guided **precision landing** (ArduPilot PrecLand) on a
Raspberry Pi companion computer, in the IFSC's CCD and CBR competition stack,
but nothing here is tied to that stack: a model file, OpenCV and either `ultralytics`
or `tflite-runtime` are enough.

```
base_pouso/                                        # task: landing base ("base de pouso")
  yolo11n/640_ncnn_model/                          # ncnn, imgsz 640, fp16      ~5 MB   <- runtime
  yolo11m/yolo11m_pose_1280.pt                     # PyTorch, imgsz 1280        ~40 MB  <- teacher
          yolo11m_pose_1280_last.pt                # last epoch (same run)      ~40 MB
          yolo11m_pose_1280_ncnn_model/            # ncnn export of the above   ~80 MB
  mobilenetv2/mobilenetv2_heatmap_pose_fp16.tflite # TFLite fp16, 640x640       ~15 MB
              mobilenetv2_pose_fp16.tflite         # TFLite fp16, 224x224       ~5 MB
pacote/                                            # reserved: future "package" task
```

## What these models are

All four solve the **same task with the same output convention**: find the landing base
and return **5 keypoints**.

| slot | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| meaning | center | top-left | top-right | bottom-right | bottom-left |

Single class: `0: base_pouso`. Corner slots are assigned by each corner's angle around
the center, so the order is stable across images (and `flip_idx: [0, 2, 1, 4, 3]` keeps
it correct under horizontal-flip augmentation).

| model | arch | input | format | output | detection conf? |
|---|---|---|---|---|---|
| `yolo11n/640_ncnn_model` | YOLO11n-pose | 640×640 | ncnn (fp16) | boxes + 5 kpts in **frame pixels** | yes (box + per-kpt) |
| `yolo11m/yolo11m_pose_1280.pt` | YOLO11m-pose | 1280×1280 | PyTorch (+ncnn) | boxes + 5 kpts in **frame pixels** | yes (box + per-kpt) |
| `mobilenetv2/..._heatmap_pose_fp16.tflite` | MobileNetV2 + 2× Conv2DTranspose → 80×80×5 heatmaps → soft-argmax | 640×640 | TFLite fp16 | `(1, 5, 2)`, **normalized [0,1]** | **no** |
| `mobilenetv2/mobilenetv2_pose_fp16.tflite` | MobileNetV2 + GAP → Dense regression | 224×224 | TFLite fp16 | `(1, 10)`, **normalized [0,1]** | **no** |

The two MobileNetV2 files have **no detector head**: they always return 5 points, even on
a frame with no base in it — you must add your own plausibility check. Decoding
(soft-argmax / sigmoid) is baked into the exported graph, so the output is already
coordinates; there is no heatmap post-processing to write.

## Why

A landing controller needs a target direction every frame. The **center** keypoint gives
that; the **4 corners** additionally give scale (→ a rough altitude estimate,
`Z ≈ ALT_SCALE / d_px`) and the yaw of the base — neither of which a plain bbox detector
can provide, and without needing an ArUco/fiducial marker painted on the base.

Why four models instead of one:

- **`yolo11n` @640 in ncnn — the deployment model.** ncnn is the only format that keeps a
  usable frame rate on a Raspberry Pi, and the nano variant fits the CPU budget.
  Benchmarked head-to-head at the same 640 input (16-core x86 container, 50 iterations of
  synthetic input, best over 4/8/16 threads): **11.9 FPS** vs **2.4 FPS** for the
  MobileNetV2 heatmap model — the "lighter" MobileNet is ~5× *slower*, because its
  deconv + soft-argmax head does not parallelize under XNNPACK. Cost: ~630 MB RAM, since
  torch + ultralytics get pulled in.
- **`yolo11m` @1280 — the teacher, not for flight.** Best accuracy on small/distant
  corners. Used offline to pre-label new footage (model-in-the-loop annotation) and as the
  quality ceiling to compare the small models against.
- **`mobilenetv2` heatmap @640 — the lightweight alternative that lost.** Kept for
  reproducibility and as a `tflite-runtime`-only fallback: no torch, no ultralytics,
  ~231 MB RAM. It would need a 320 input and numpy-side decoding to be competitive.
- **`mobilenetv2` regression @224 — the first baseline.** Global average pooling throws
  away spatial information and biases predictions toward the image center; superseded by
  the heatmap variant. Useful only as a very small, very fast smoke test.

## What they were trained on

**Dataset.** Frames recorded from the drone's own downward camera, annotated in
**Label Studio** with two keypoint labels — `center` and `corner` — plus a rotated
rectangle for the bbox. Exported as **native Label Studio JSON**, not COCO and not the LS
"YOLO" export: LS cannot represent 4 corners sharing a single label and silently drops
every keypoint, so the JSON export is converted to YOLO-pose format by a custom script.
Staged on Kaggle as `yabrick/base-pouso-pose-ls91` (raw LS JSON) and
`yabrick/base-pouso-yolo-pose` (already converted). Corners missing from an annotation are
written as `0 0 0` (visibility 0) and ignored by the loss, so partially visible bases are
still usable training samples.

**Training** (all on Kaggle GPUs):

| model | recipe |
|---|---|
| `yolo11n` 640 | from `yolo11n-pose.pt` (COCO-pretrained), 220 epochs, batch 64, 2×T4, `lr0=0.001 lrf=0.01`, patience 50, split 70/30 |
| `yolo11m` 1280 | from a COCO-pretrained pose checkpoint, imgsz 1280, small batch on a single P100, split 80/20 |
| MobileNetV2 heatmap | ImageNet backbone; Gaussian-blob targets (σ=3 on the 80×80 grid), masked MSE; phase 1 frozen backbone 40 ep, phase 2 fine-tune 130 ep @1e-4; split 80/20 |
| MobileNetV2 regression | ImageNet backbone; masked MSE on normalized coords; 40 ep frozen + 120 ep fine-tune @1e-4; split 80/20 |

Augmentation follows the same idea everywhere: HSV/brightness jitter, ±10–15° rotation,
±30–50% scale, horizontal flip (with the TL↔TR / BL↔BR slot swap) and a light vertical flip.

Metrics differ by family: the YOLO models report `mAP50 / mAP50-95` for box **and** pose;
the MobileNetV2 models are regressors with no detection score, so they report mean
normalized keypoint error, PCK@0.05 and an OKS-AP@0.5 proxy.

Export toolchain: `.pt → ncnn` via **pnnx** (`model.export(format='ncnn', half=True)`),
`.keras → TFLite` via `tf.lite.TFLiteConverter` with `supported_types=[tf.float16]`.
**Never convert from the ncnn folder — always export from the original `.pt` / `.keras`.**

## Usage examples

### YOLO11-pose (ncnn) — the runtime path

```python
import cv2
import numpy as np
from ultralytics import YOLO

# Point at the ncnn *directory*, not a file. task="pose" is required for ncnn.
model = YOLO("base_pouso/yolo11n/640_ncnn_model", task="pose")

frame = cv2.imread("frame.jpg")                       # BGR, any resolution
res = model(np.ascontiguousarray(frame), imgsz=640, conf=0.8, verbose=False)[0]

if len(res.boxes) == 0:
    print("no base in frame")
else:
    i = int(res.boxes.conf.argmax())                  # best instance
    kps = res.keypoints.xy[i].cpu().numpy()           # (5, 2) in ORIGINAL frame pixels
    kconf = res.keypoints.conf[i].cpu().numpy()       # (5,) per-keypoint confidence
    center, corners = kps[0], kps[1:]                 # corners = TL, TR, BR, BL

    # Altitude proxy: d_px = f_px * d_real / Z  ->  Z = ALT_SCALE / d_px
    # Calibrate once at a known height: ALT_SCALE = Z_known * d_px_measured.
    ALT_SCALE = 250.0
    d = float(np.linalg.norm(center - corners[0]))
    alt = np.clip(ALT_SCALE / d, 0.0, 6.0) if d >= 10 else None

    # Base yaw, from center -> TL corner
    yaw_deg = (np.degrees(np.arctan2(*(corners[0] - center)[::-1])) - 90) % 360

    print(f"center={center}  alt~{alt}  yaw={yaw_deg:.1f} deg")
```

The ncnn export returns coordinates already mapped back to the original frame — no manual
letterbox undo needed.

### YOLO11m @1280 (.pt) — offline / pre-labeling

```python
from ultralytics import YOLO

model = YOLO("base_pouso/yolo11m/yolo11m_pose_1280.pt")

# stream=True keeps memory flat over long folders
for res in model.predict("images/", imgsz=1280, conf=0.5, stream=True, verbose=False):
    if len(res.boxes) == 0:
        continue
    i = int(res.boxes.conf.argmax())
    print(res.path, res.keypoints.xy[i].cpu().numpy()[0])   # file, base center

# Re-export the runtime formats:
# model.export(format="ncnn",   imgsz=1280, half=True)
# model.export(format="tflite", imgsz=1280)
```

### MobileNetV2 (TFLite) — both variants, one loader

The input size is read from the file, so the same code drives the 640×640 heatmap model
and the 224×224 regression model.

```python
import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter      # light runtime (RPi / edge)
except ImportError:
    from tensorflow.lite import Interpreter                 # desktop fallback

MODEL = "base_pouso/mobilenetv2/mobilenetv2_heatmap_pose_fp16.tflite"
# MODEL = "base_pouso/mobilenetv2/mobilenetv2_pose_fp16.tflite"   # 224px, same code

itp = Interpreter(model_path=MODEL, num_threads=4)          # ~physical cores; more oversubscribes
itp.allocate_tensors()
inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
IH, IW = int(inp["shape"][1]), int(inp["shape"][2])          # 640x640 or 224x224

def predict(frame):
    """frame: BGR uint8 -> (5, 2) keypoints in frame pixels."""
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (IW, IH))                       # aspect squashed, as in training
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img / 127.5 - 1.0                                 # mobilenet_v2 preprocess_input
    itp.set_tensor(inp["index"], img[None].astype(inp["dtype"]))
    itp.invoke()
    kps_n = itp.get_tensor(out["index"])[0].reshape(-1, 2)  # (5, 2) in [0, 1]
    return kps_n * np.array([w, h], dtype=np.float32)

kps = predict(cv2.imread("frame.jpg"))
center, corners = kps[0], kps[1:]

# No confidence output -> add your own sanity gate:
if np.linalg.norm(center - corners[0]) < 10:                # degenerate
    print("treat as: no base")
```

### Minimal ROS node

Same inference, wrapped so the base center and the altitude estimate become a
`geometry_msgs/Point` other nodes (e.g. a precision-landing sender) can subscribe to:

```python
#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from ultralytics import YOLO

ALT_SCALE = 250.0          # px*m, calibrate: ALT_SCALE = Z_known * d_px_measured

class BaseReader:
    def __init__(self):
        self.model = YOLO(rospy.get_param("~model"), task="pose")
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("base_reader/pose", Point, queue_size=1)
        rospy.Subscriber("/camera/image_raw", Image, self.on_frame, queue_size=1)

    def on_frame(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        res = self.model(np.ascontiguousarray(frame), imgsz=640, conf=0.8, verbose=False)[0]
        if len(res.boxes) == 0:
            return
        i = int(res.boxes.conf.argmax())
        kps = res.keypoints.xy[i].cpu().numpy()
        center, corner = kps[0], kps[1]
        d = float(np.linalg.norm(center - corner))
        if d < 10:                                   # too small / degenerate -> ignore
            return
        self.pub.publish(Point(float(center[0]), float(center[1]),
                               float(np.clip(ALT_SCALE / d, 0.0, 6.0))))

if __name__ == "__main__":
    rospy.init_node("base_reader")
    BaseReader()
    rospy.spin()
```

### Requirements

```bash
pip install ultralytics       # YOLO models; ncnn runtime ships with it
pip install tflite-runtime    # MobileNetV2 models on edge (or: pip install tensorflow)
```
