#!/usr/bin/env python3
# Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT

"""
SpinalML MNIST Interactive Demo with Tang Primer 20K Hardware Inference.
Allows drawing digits on a canvas and classifying them in real-time
either via physical FPGA hardware over UART (W4A8 architecture) or via the local NumPy replica.
"""

import sys
import time
import math
import struct
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    import gradio as gr
except ImportError:
    print("[ERROR] Gradio is required. Run 'pip install -r requirements.txt'")
    sys.exit(1)

# Protocol opcodes matching UartBridge & docs/uart_bridge.md
CMD_CSR_WRITE   = 0x43  # 'C'
CMD_MEM_WRITE   = 0x57  # 'W'
CMD_READ_LOGITS = 0x52  # 'R'
CMD_STATUS      = 0x53  # 'S'
CMD_VERSION     = 0x56  # 'V'

DEFAULT_PORT = "COM8" if sys.platform == "win32" else "/dev/ttyUSB0"
DEFAULT_BAUD = 115200

IMG_BASE_ADDR    = 0x10000  # Virtual BRAM index 0
WEIGHT_BASE_ADDR = 0x20000  # Virtual BRAM index 2048
NUM_CLASSES      = 10

# State tracking for uploaded weights
_WEIGHTS_LOADED_PORT = None

# Load pre-quantized weights from Mnist_weights.npz
WEIGHTS_FILE = Path(__file__).parent / "Mnist_weights.npz"
MODEL_DATA = None
HARDWARE_PAYLOAD = None

if WEIGHTS_FILE.exists():
    try:
        data = np.load(WEIGHTS_FILE)
        MODEL_DATA = {
            "convWq": data["convWq"],
            "convBq": data["convBq"],
            "fcW": data["fcW"],
            "fcB": data["fcB"],
            "convScale": float(data["convScale"]),
        }
        if "hardware_payload" in data:
            HARDWARE_PAYLOAD = bytes(data["hardware_payload"])
        print(f"[INFO] Loaded model weights from {WEIGHTS_FILE.name} (hardware payload: {len(HARDWARE_PAYLOAD) if HARDWARE_PAYLOAD else 0} bytes)")
    except Exception as e:
        print(f"[WARNING] Could not load local weights: {e}")


def encode_e4m3(f: float) -> int:
    """Encodes a float to 8-bit FP8 (E4M3: 1 sign, 4 exp, 3 mantissa, bias 7)."""
    if f == 0.0:
        return 0
    sign = 0x80 if f < 0 else 0
    a = abs(float(f))
    if a >= (2.0 ** -6):
        e = int(math.floor(math.log2(a)))
        mF = a / (2.0 ** e)
        if mF >= 2.0:
            mF /= 2.0
            e += 1
        m = int(math.floor((mF - 1) * 8 + 0.5))
        if m == 8:
            m = 0
            e += 1
        eb = e + 7
        return sign | (eb << 3) | m
    else:
        m = int(math.floor(a * 512 + 0.5))
        return sign | m


def decode_e4m3(b: int) -> float:
    """Decodes an 8-bit FP8 (E4M3) byte to float."""
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 3) & 0x0F
    mant = b & 0x07
    if exp == 0:
        val = (mant / 8.0) * (2.0 ** -6)
    else:
        val = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
    return sign * val


def softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax computation."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess_canvas_image(image_input):
    """
    Extracts, centers, and binarizes drawn canvas strokes to a 28x28 grid.
    Returns:
        canvas_28: (28, 28) uint8 array with values in {0, 1}
        raw_bytes: 784 bytes (0x00 or 0x01) formatted for FPGA upload
    """
    if image_input is None:
        return None, None

    if isinstance(image_input, dict):
        img_arr = image_input.get("composite", image_input.get("background"))
    else:
        img_arr = image_input

    if img_arr is None:
        return None, None

    if isinstance(img_arr, np.ndarray):
        if len(img_arr.shape) == 3:
            if img_arr.shape[2] == 4:
                # Use alpha channel if background is transparent
                alpha = img_arr[:, :, 3]
                if np.max(alpha) > 0 and np.min(alpha) < 255:
                    pil_img = Image.fromarray(alpha)
                else:
                    pil_img = Image.fromarray(img_arr[:, :, :3]).convert("L")
            else:
                pil_img = Image.fromarray(img_arr).convert("L")
        else:
            pil_img = Image.fromarray(img_arr)
    else:
        pil_img = image_input.convert("L")

    arr = np.array(pil_img, dtype=np.float32)

    # Invert colors if needed (digit drawn in dark on light background)
    if np.mean(arr) > 127:
        arr = 255.0 - arr

    if np.max(arr) < 20:
        return None, None

    # Binarize strokes (white digits > 40 on black background)
    binary = (arr > 40).astype(np.uint8)

    # Center bounding box in a 20x20 area within 28x28 canvas (MNIST standard)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None, None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    # Crop the grayscale image directly with bounding box
    crop = arr[ymin:ymax+1, xmin:xmax+1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Downscale grayscale with antialiasing (LANCZOS)
    crop_pil = Image.fromarray(crop.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.LANCZOS)
    # Threshold at > 35 to ensure unbroken solid stroke thickness (2-3 pixels wide in 28x28)
    crop_resized = (np.array(crop_pil) > 35).astype(np.uint8)

    canvas_28 = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas_28[y_off:y_off+new_h, x_off:x_off+new_w] = crop_resized

    # Ensure solid stroke thickness (2-3 pixels wide for MNIST)
    if np.sum(canvas_28) < 90:
        dilated = canvas_28.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(canvas_28, dy, axis=0), dx, axis=1)
                dilated = np.maximum(dilated, shifted)
        canvas_28 = dilated

    raw_bytes = canvas_28.tobytes()
    return canvas_28, raw_bytes


def upload_memory_bytes(ser, base_addr: int, data: bytes):
    """
    Sends data into FPGA BRAM via UART 'W' command, chunked in 256-byte blocks.
    """
    length = len(data)
    header = bytes([CMD_MEM_WRITE]) + struct.pack("<II", base_addr, length)
    ser.write(header)
    ser.flush()

    chunk_size = 256
    for i in range(0, length, chunk_size):
        chunk = data[i:i+chunk_size]
        ser.write(chunk)
        ser.flush()
        time.sleep(0.002)


def ensure_weights_uploaded(ser, port: str):
    """Ensures model weights are loaded to FPGA BRAM at 0x20000."""
    global _WEIGHTS_LOADED_PORT
    if _WEIGHTS_LOADED_PORT == port:
        return False  # Already loaded

    if HARDWARE_PAYLOAD is None:
        raise RuntimeError("Hardware weight payload not found in Mnist_weights.npz.")

    upload_memory_bytes(ser, WEIGHT_BASE_ADDR, HARDWARE_PAYLOAD)
    _WEIGHTS_LOADED_PORT = port
    return True


def infer_on_hardware(raw_bytes: bytes, port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD):
    """
    Transmits input image to Tang Primer 20K FPGA over UART and reads back 10 FP8 logits.
    """
    if not HAS_SERIAL:
        raise RuntimeError("pyserial is not installed.")

    ser = serial.Serial(port=port, baudrate=baud, timeout=1.0, write_timeout=1.0)
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # 1. Ping Protocol Version ('V')
        ser.write(bytes([CMD_VERSION]))
        ser.flush()
        v_resp = ser.read(1)
        if len(v_resp) != 1 or v_resp[0] != 0x01:
            raise ConnectionError(f"FPGA not responding with protocol v1 on {port}")

        # 2. Ensure weights are present at 0x20000
        weights_loaded_just_now = ensure_weights_uploaded(ser, port)

        # 3. Upload 28x28 Input Image (784 bytes) to BRAM at 0x10000 ('W')
        upload_memory_bytes(ser, IMG_BASE_ADDR, raw_bytes)

        # 4. Configure CSRs
        # CSR 0x08: Image Base Address
        ser.write(bytes([CMD_CSR_WRITE]) + struct.pack("<II", 0x08, IMG_BASE_ADDR))
        ser.flush()
        time.sleep(0.002)

        # CSR 0x0C: Weights Base Address
        ser.write(bytes([CMD_CSR_WRITE]) + struct.pack("<II", 0x0C, WEIGHT_BASE_ADDR))
        ser.flush()
        time.sleep(0.002)

        # CSR 0x00: Trigger Inference (start pulse)
        ser.write(bytes([CMD_CSR_WRITE]) + struct.pack("<II", 0x00, 0x01))
        ser.flush()

        # Wait for inference completion (~0.3 ms at 27 MHz)
        time.sleep(0.02)

        # 5. Read 10 Output Logits ('R')
        ser.write(bytes([CMD_READ_LOGITS]))
        ser.flush()
        logits_raw = ser.read(NUM_CLASSES)
        if len(logits_raw) != NUM_CLASSES:
            raise TimeoutError(f"Expected {NUM_CLASSES} bytes, received {len(logits_raw)}")

        # Decode FP8 (E4M3) logits
        logits = np.array([decode_e4m3(b) for b in logits_raw], dtype=np.float32)
        return logits, weights_loaded_just_now
    finally:
        ser.close()


def infer_locally_numpy(pix_28x28: np.ndarray) -> np.ndarray:
    """
    Bit-exact software replica of the W4A8 hardware accelerator forward pass.
    """
    if MODEL_DATA is None:
        return np.zeros(NUM_CLASSES, dtype=np.float32)

    convWq = MODEL_DATA["convWq"]
    convBq = MODEL_DATA["convBq"]
    fcW = MODEL_DATA["fcW"]
    fcB = MODEL_DATA["fcB"]
    convScale = MODEL_DATA["convScale"]

    # 1. Conv2D (1 -> 2 channels, 5x5 valid pad, I16 accum)
    conv_out = np.zeros((2, 24, 24), dtype=np.int32)
    for c in range(2):
        w = convWq[c].reshape(5, 5)
        bias = int(convBq[c])
        for y in range(24):
            for x in range(24):
                patch = pix_28x28[y:y+5, x:x+5]
                acc = int(np.sum(patch * w)) + bias
                conv_out[c, y, x] = max(acc, 0)  # ReLU

    # 2. MaxPool2D (2x2, stride 2) -> (2, 12, 12), Cast to FP8 E4M3 in features-last order
    acts = []
    for i in range(12):
        for j in range(12):
            for c in range(2):
                v = int(np.max(conv_out[c, 2*i:2*i+2, 2*j:2*j+2]))
                f_val = decode_e4m3(encode_e4m3(v * convScale))
                acts.append(f_val)
    acts = np.array(acts, dtype=np.float32)

    # 3. Linear (288 -> 10) in FP8 domain
    w_mat = np.array([[decode_e4m3(encode_e4m3(x)) for x in row] for row in fcW], dtype=np.float32)
    b_vec = np.array([decode_e4m3(encode_e4m3(x)) for x in fcB], dtype=np.float32)
    logits = w_mat @ acts + b_vec
    return logits


def classify_digit(image_input, port_name):
    """
    Main prediction pipeline called when the drawing canvas changes.
    Runs both FPGA Hardware and NumPy Software replica side-by-side.
    """
    pix_28x28, raw_bytes = preprocess_canvas_image(image_input)
    empty_dist = {str(i): 0.0 for i in range(10)}

    if pix_28x28 is None or raw_bytes is None:
        return (
            empty_dist, "Awaiting input...",
            empty_dist, "Awaiting input..."
        )

    # 1. Software Reference Inference (NumPy Golden Replica)
    t0_sw = time.perf_counter()
    sw_logits = infer_locally_numpy(pix_28x28)
    sw_time_ms = (time.perf_counter() - t0_sw) * 1000
    sw_probs = softmax(sw_logits)
    sw_pred = int(np.argmax(sw_probs))
    sw_dict = {str(i): float(sw_probs[i]) for i in range(10)}
    sw_info = f"Class {sw_pred} ({sw_probs[sw_pred]*100:.1f}%) | Latency: {sw_time_ms:.2f}ms"

    # 2. Physical Hardware Inference (Tang Primer 20K FPGA over UART)
    hw_pred = None
    t0_hw = time.perf_counter()
    try:
        hw_logits, weights_uploaded = infer_on_hardware(raw_bytes, port=port_name)
        hw_time_ms = (time.perf_counter() - t0_hw) * 1000
        hw_probs = softmax(hw_logits)
        hw_pred = int(np.argmax(hw_probs))
        hw_dict = {str(i): float(hw_probs[i]) for i in range(10)}
        load_tag = " [Weights Auto-Loaded]" if weights_uploaded else ""
        hw_info = f"Class {hw_pred} ({hw_probs[hw_pred]*100:.1f}%) | RTT: {hw_time_ms:.1f}ms{load_tag}"
    except Exception as e:
        hw_dict = empty_dist
        hw_info = f"FPGA Offline ({e})"

    return hw_dict, hw_info, sw_dict, sw_info


def test_or_initialize_fpga(port_name):
    """Pings the FPGA UART bridge and preloads weights."""
    global _WEIGHTS_LOADED_PORT
    if not HAS_SERIAL:
        return "pyserial not installed."
    try:
        ser = serial.Serial(port=port_name, baudrate=DEFAULT_BAUD, timeout=1.5, write_timeout=1.5)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # Ping 'V'
        ser.write(bytes([CMD_VERSION]))
        ser.flush()
        resp = ser.read(1)
        if len(resp) != 1 or resp[0] != 0x01:
            ser.close()
            return f"UNEXPECTED: Ping returned 0x{resp.hex() if resp else 'TIMEOUT'} (expected 0x01)"

        # Check status 'S'
        ser.write(bytes([CMD_STATUS]))
        ser.flush()
        st_resp = ser.read(1)
        st_byte = st_resp[0] if st_resp else 0

        # Upload weights
        if HARDWARE_PAYLOAD is not None:
            upload_memory_bytes(ser, WEIGHT_BASE_ADDR, HARDWARE_PAYLOAD)
            _WEIGHTS_LOADED_PORT = port_name
            ser.close()
            return f"CONNECTED & READY: Tang Primer 20K on {port_name} (Status: 0x{st_byte:02X}, Weights: {len(HARDWARE_PAYLOAD)}B loaded at 0x20000)"
        else:
            ser.close()
            return f"CONNECTED: Tang Primer 20K on {port_name} (Warning: hardware payload missing)"
    except Exception as e:
        return f"FAILED: {e}"


CUSTOM_CSS = """
/* Lock pen width and hide the brush size slider toolbar */
input[type="range"], .brush-size, [aria-label="Brush size"] {
    display: none !important;
}
"""

# --- Gradio User Interface ---
with gr.Blocks(title="SpinalML Tang Primer 20K MNIST Demo") as demo:
    gr.Markdown("# SpinalML: Real-Time MNIST Hardware vs Software Inference")
    gr.Markdown(
        "Draw a digit from **0 to 9** on the canvas. "
        "The system simultaneously runs the **physical Tang Primer 20K FPGA** accelerator over UART "
        "and the **NumPy reference** side-by-side for live in-circuit verification."
    )

    with gr.Row():
        # Column 1: Canvas & Hardware Connection Controls
        with gr.Column(scale=4):
            canvas = gr.Sketchpad(
                label="Digit Canvas (280x280)",
                type="numpy",
                image_mode="L",
                canvas_size=(280, 280),
                brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=12),
            )

            gr.Markdown("#### FPGA Hardware Link")
            with gr.Row():
                port_input = gr.Textbox(label="UART Port", value=DEFAULT_PORT, scale=1)
                init_btn = gr.Button("Connect & Load Weights", variant="primary", scale=1)

        # Column 2: Side-by-Side FPGA vs NumPy outputs + Link Status below
        with gr.Column(scale=7):
            with gr.Row():
                # Sub-column: Tang Primer 20K Physical Hardware Output
                with gr.Column(scale=1):
                    gr.Markdown("### Tang Primer 20K (Physical FPGA)")
                    hw_label = gr.Label(label="FPGA Confidence Scores", num_top_classes=3)
                    hw_info = gr.Textbox(label="Hardware Telemetry", interactive=False)

                # Sub-column: NumPy Reference Output
                with gr.Column(scale=1):
                    gr.Markdown("### NumPy Reference (Software Golden)")
                    sw_label = gr.Label(label="Software Confidence Scores", num_top_classes=3)
                    sw_info = gr.Textbox(label="Software Telemetry", interactive=False)

            conn_status = gr.Textbox(label="Link Status", interactive=False)

    gr.Markdown(
        "<div style='text-align: center; color: #6b7280; font-size: 0.8rem; margin-top: 1.5rem;'>"
        "Copyright (c) 2026 Léonard Adamo (Juste-Leo2) - SPDX-License-Identifier: MIT"
        "</div>"
    )

    # Event handlers
    canvas.change(
        fn=classify_digit,
        inputs=[canvas, port_input],
        outputs=[hw_label, hw_info, sw_label, sw_info]
    )
    init_btn.click(fn=test_or_initialize_fpga, inputs=[port_input], outputs=[conn_status])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=CUSTOM_CSS)
