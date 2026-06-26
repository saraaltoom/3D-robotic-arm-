import cv2
import numpy as np
import serial
import time
import threading
import json
import queue
from enum import Enum, auto

import whisper
import sounddevice as sd
import groq

# ─────────────────────────────────────────────
#  CONFIG — ضعي مفتاحك هنا
# ─────────────────────────────────────────────
GROQ_API_KEY   = ""   
GROQ_MODEL     = "llama-3.3-70b-versatile"

COM_PORT       = "COM7"
BAUD_RATE      = 9600
SERIAL_TIMEOUT = 0.5
STEP_DELAY     = 1.2

SAMPLE_RATE    = 16000
RECORD_SECS    = 4        # ثواني التسجيل عند الضغط على SPACE

# ─────────────────────────────────────────────
#  MOTOR SEQUENCES
# ─────────────────────────────────────────────
NORMAL_POS = [("a", 90), ("b", 160), ("c", 100), ("d", 35)]

SEQUENCES = {
    "pickup": [
        ("a", 90), ("b", 120), ("c", 150), ("d", 15)
    ],
    "transfer": [
        ("b", 160), ("c", 100), ("a", 50),
        ("b", 120), ("c", 150), ("d", 35)
    ],
    "return_home": [
        ("a", 90), ("b", 160), ("c", 100), ("d", 35)
    ],
    "open_gripper":  [("d", 15)],
    "close_gripper": [("d", 35)],
    "move_left":     [("a", 50),  ("b", 140), ("c", 110)],
    "move_right":    [("a", 130), ("b", 140), ("c", 110)],
    "move_forward":  [("b", 110), ("c", 160)],
    "move_back":     [("b", 160), ("c", 100)],
}

# ─────────────────────────────────────────────
#  YELLOW DETECTION
# ─────────────────────────────────────────────
YELLOW_LOWER    = np.array([22, 150, 150])
YELLOW_UPPER    = np.array([33, 255, 255])
MIN_YELLOW_AREA = 2500
CONFIRM_FRAMES  = 10
RETURN_ZONE_PX  = 60

# ─────────────────────────────────────────────
#  SYSTEM PROMPT للـ LLM
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are the brain of a 6-DOF robot arm with a camera.
Read a natural-language command and return ONLY a JSON object.

Available actions:
- "pickup"        : pick up the yellow object
- "transfer"      : move it to the drop zone
- "return_home"   : go back to home position
- "open_gripper"  : open the gripper
- "close_gripper" : close the gripper
- "move_left"     : rotate arm left
- "move_right"    : rotate arm right
- "move_forward"  : extend arm forward
- "move_back"     : retract arm back
- "full_cycle"    : pick + transfer + return (complete task)
- "stop"          : stop immediately
- "unknown"       : command unclear

Rules:
- "make coffee / prepare coffee / bring the cup" → "full_cycle"
- "grab / pick / take / get" → "pickup"
- "move / transfer / put it there" → "transfer"
- "go home / reset / normal position" → "return_home"
- "open / release / let go" → "open_gripper"
- "close / grip / hold" → "close_gripper"
- "left" → "move_left" | "right" → "move_right"
- "forward / extend" → "move_forward" | "back / retract" → "move_back"
- "stop / halt / freeze" → "stop"

Return ONLY this JSON, nothing else:
{
  "action": "<action_key>",
  "confidence": <0.0 to 1.0>,
  "explanation": "<one short sentence>"
}
"""

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
class State(Enum):
    IDLE         = auto()
    PICKING      = auto()
    TRANSFERRING = auto()
    RETURNING    = auto()
    OBJECT_MOVED = auto()
    STOPPED      = auto()

# ─────────────────────────────────────────────
#  SERIAL
# ─────────────────────────────────────────────
def open_serial(port):
    try:
        s = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        time.sleep(2.5)
        s.reset_input_buffer()
        return s
    except Exception as e:
        print(f"[Serial] {e}  →  simulation mode")
        return None

def send_cmd(ser, motor, angle):
    cmd = f"{motor}{angle}"
    print(f"  CMD → {cmd}")
    if ser:
        try:
            ser.write((cmd + "\n").encode())
            ser.flush()
            time.sleep(0.05)
        except Exception as e:
            print(f"  [Serial write] {e}")

# ─────────────────────────────────────────────
#  VISION
# ─────────────────────────────────────────────
def clean_mask(mask, k=9, it=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=it)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=it)
    return mask

def detect_yellow(frame, lower, upper):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = clean_mask(mask, k=9, it=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, 0, None
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < MIN_YELLOW_AREA:
        return False, None, 0, None
    M = cv2.moments(best)
    if M["m00"] == 0:
        return False, None, 0, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return True, (cx, cy), int(area), best

def sample_hsv(frame, x, y, r=12):
    h, w  = frame.shape[:2]
    patch = cv2.cvtColor(
        frame[max(0,y-r):min(h,y+r), max(0,x-r):min(w,x+r)],
        cv2.COLOR_BGR2HSV
    )
    return patch.reshape(-1, 3).mean(axis=0).astype(int)

# ─────────────────────────────────────────────
#  ROBOT CONTROLLER
# ─────────────────────────────────────────────
class RobotController:
    def __init__(self, ser):
        self.ser   = ser
        self._state = State.IDLE
        self._lock  = threading.Lock()
        self.original_centroid = None
        self.busy  = False
        self.last_action      = ""
        self.last_explanation = ""

    @property
    def state(self):
        with self._lock:
            return self._state

    def set_state(self, s):
        with self._lock:
            self._state = s
        print(f"[STATE] → {s.name}")

    def reset(self):
        with self._lock:
            self._state = State.IDLE
            self.original_centroid = None
            self.busy = False
        print("[RESET] → IDLE")

    def _run_seq(self, seq):
        for motor, angle in seq:
            if self.state == State.STOPPED:
                print("[STOP] sequence aborted")
                return False
            send_cmd(self.ser, motor, angle)
            time.sleep(STEP_DELAY)
        return True

    def _execute(self, action):
        try:
            if action == "pickup":
                self.set_state(State.PICKING)
                self._run_seq(SEQUENCES["pickup"])

            elif action == "transfer":
                self.set_state(State.TRANSFERRING)
                self._run_seq(SEQUENCES["transfer"])

            elif action == "return_home":
                self.set_state(State.RETURNING)
                self._run_seq(SEQUENCES["return_home"])
                self.set_state(State.IDLE)

            elif action == "full_cycle":
                self.set_state(State.PICKING)
                if not self._run_seq(SEQUENCES["pickup"]):    return
                self.set_state(State.TRANSFERRING)
                if not self._run_seq(SEQUENCES["transfer"]):  return
                self.set_state(State.RETURNING)
                if not self._run_seq(SEQUENCES["return_home"]): return
                self.set_state(State.OBJECT_MOVED)

            elif action in ("open_gripper", "close_gripper",
                            "move_left", "move_right",
                            "move_forward", "move_back"):
                self._run_seq(SEQUENCES[action])

            elif action == "stop":
                self.set_state(State.STOPPED)

            else:
                print(f"[NLP] Unknown action: {action}")

        finally:
            with self._lock:
                if self._state not in (State.STOPPED, State.OBJECT_MOVED):
                    self._state = State.IDLE
                self.busy = False

    def run_action(self, action, explanation=""):
        with self._lock:
            if self.busy:
                print("[BUSY] arm is already moving")
                return
            if self._state == State.STOPPED and action != "return_home":
                print("[STOPPED] say 'go home' first to resume")
                return
            self.busy = True
            self.last_action      = action
            self.last_explanation = explanation

        threading.Thread(target=self._execute, args=(action,), daemon=True).start()

    def notify_yellow_seen(self, centroid):
        oc = self.original_centroid
        if oc is None:
            return
        dist = ((centroid[0]-oc[0])**2 + (centroid[1]-oc[1])**2) ** 0.5
        if dist <= RETURN_ZONE_PX:
            self.original_centroid = None
            self.set_state(State.IDLE)

# ─────────────────────────────────────────────
#  SPEECH → TEXT  (Whisper local — مجاني)
# ─────────────────────────────────────────────
class SpeechListener:
    def __init__(self):
        print("[Whisper] Loading tiny model …")
        self.model     = whisper.load_model("tiny")
        self.cmd_queue = queue.Queue()
        self.recording = False
        print("[Whisper] Ready ✓")

    def _record_and_transcribe(self):
        self.recording = True
        print(f"[MIC] Recording {RECORD_SECS}s …")
        audio = sd.rec(
            int(RECORD_SECS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        self.recording = False
        text = self.model.transcribe(audio.flatten(), language="en", fp16=False)["text"].strip()
        print(f'[STT] "{text}"')
        if text:
            self.cmd_queue.put(text)

    def listen_async(self):
        if not self.recording:
            threading.Thread(target=self._record_and_transcribe, daemon=True).start()

# ─────────────────────────────────────────────
#  NLP  (Groq — مجاني)
# ─────────────────────────────────────────────
class NLPProcessor:
    def __init__(self, api_key):
        self.client = groq.Groq(api_key=api_key)

    def parse(self, text, scene_info=""):
        user_msg = f"Command: {text}"
        if scene_info:
            user_msg += f"\nScene: {scene_info}"
        try:
            resp = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg}
                ],
                temperature=0.1,
                max_tokens=150
            )
            raw  = resp.choices[0].message.content.strip()
            raw  = raw.replace("```json","").replace("```","").strip()
            data = json.loads(raw)
            return (
                data.get("action", "unknown"),
                data.get("explanation", ""),
                data.get("confidence", 0.0)
            )
        except Exception as e:
            print(f"[Groq] Error: {e}")
            return "unknown", str(e), 0.0

# ─────────────────────────────────────────────
#  CALIBRATION MOUSE
# ─────────────────────────────────────────────
calib_mode  = False
calib_frame = None
calib_lower = YELLOW_LOWER.copy()
calib_upper = YELLOW_UPPER.copy()

def mouse_cb(event, x, y, flags, param):
    global calib_mode, calib_frame, calib_lower, calib_upper
    if event == cv2.EVENT_LBUTTONDOWN and calib_mode and calib_frame is not None:
        hsv = sample_hsv(calib_frame, x, y, r=14)
        H, S, V = int(hsv[0]), int(hsv[1]), int(hsv[2])
        calib_lower = np.array([max(0,H-6),   max(0,S-60), max(0,V-60)])
        calib_upper = np.array([min(179,H+6), 255,         255        ])
        print(f"[CALIB] HSV=({H},{S},{V})  lower={calib_lower}  upper={calib_upper}")
        calib_mode = False

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    global calib_mode, calib_frame, calib_lower, calib_upper

    print("=" * 55)
    print("  AI ROBOT ARM — Voice + Vision Control")
    print("=" * 55)

    ser   = open_serial(COM_PORT)
    robot = RobotController(ser)
    stt   = SpeechListener()
    nlp   = NLPProcessor(GROQ_API_KEY)

    print("\n[ARM] Moving to home position …")
    for motor, angle in NORMAL_POS:
        send_cmd(ser, motor, angle)
        time.sleep(0.9)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("AI Arm Controller")
    cv2.setMouseCallback("AI Arm Controller", mouse_cb)

    STATE_COLORS = {
        State.IDLE:         (0,   200,   0),
        State.PICKING:      (0,   165, 255),
        State.TRANSFERRING: (0,   100, 255),
        State.RETURNING:    (255, 165,   0),
        State.OBJECT_MOVED: (130, 130, 255),
        State.STOPPED:      (0,     0, 220),
    }

    confirm_count = 0
    status_text   = "Ready  —  press SPACE to speak a command"
    status_color  = (0, 200, 0)

    print("\nControls:")
    print("  SPACE = speak a command (4 sec recording)")
    print("  C     = calibrate yellow color")
    print("  R     = reset arm to home")
    print("  Q     = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        calib_frame = frame.copy()
        state       = robot.state

        yellow_found, centroid, area, contour = detect_yellow(
            frame, calib_lower, calib_upper
        )

        # scene description sent to LLM
        if yellow_found and centroid:
            h_f, w_f = frame.shape[:2]
            side      = "left" if centroid[0] < w_f // 2 else "right"
            scene_info = (
                f"Yellow object at ({centroid[0]},{centroid[1]}), "
                f"area={area}px², on the {side} side."
            )
        else:
            scene_info = "No yellow object visible in camera."

        # auto-confirm counter (visual feedback)
        if state == State.IDLE and not robot.busy:
            confirm_count = (confirm_count + 1) if yellow_found else max(0, confirm_count - 1)
        elif state == State.OBJECT_MOVED and yellow_found and centroid:
            robot.notify_yellow_seen(centroid)

        # ── process queued voice commands ──
        while not stt.cmd_queue.empty():
            text = stt.cmd_queue.get()
            status_text  = f'Heard: "{text}"'
            status_color = (0, 200, 255)

            action, explanation, confidence = nlp.parse(text, scene_info)
            print(f"[NLP] action={action}  conf={confidence:.2f}  → {explanation}")

            status_text  = f"{action.upper()} ({confidence:.0%}) — {explanation[:45]}"
            status_color = (0, 200, 0) if confidence > 0.6 else (0, 140, 255)
            robot.run_action(action, explanation)

        # ── draw UI ──
        h, w = frame.shape[:2]

        # top bar — state
        cv2.rectangle(frame, (0, 0), (w, 44), (20, 20, 20), -1)
        sc = STATE_COLORS.get(state, (200, 200, 200))
        cv2.putText(frame, f"STATE: {state.name}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, sc, 2, cv2.LINE_AA)

        # yellow indicator top-right
        y_txt   = "YELLOW: FOUND" if yellow_found else "YELLOW: ---"
        y_color = (0, 255, 255) if yellow_found else (80, 80, 80)
        cv2.putText(frame, y_txt, (w - 190, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, y_color, 1, cv2.LINE_AA)

        # confirm progress bar
        if state == State.IDLE and confirm_count > 0:
            bw = int(w * confirm_count / CONFIRM_FRAMES)
            cv2.rectangle(frame, (0, 44), (bw, 52), (0, 220, 0), -1)

        # detected contour & centroid
        if yellow_found and centroid:
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
            cv2.circle(frame, centroid, 10, (0, 255, 255), -1)
            cv2.putText(frame, f"{area}px²",
                        (centroid[0] + 14, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # return zone marker
        if state == State.OBJECT_MOVED and robot.original_centroid:
            ox, oy = robot.original_centroid
            cv2.circle(frame, (ox, oy), RETURN_ZONE_PX, (0, 255, 255), 2)
            cv2.drawMarker(frame, (ox, oy), (0, 255, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(frame, "Return object here",
                        (ox - 70, oy - RETURN_ZONE_PX - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

        # recording dot
        if stt.recording:
            cv2.circle(frame, (w - 22, 66), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 56, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # last NLP action info
        if robot.last_action:
            cv2.putText(frame, f"Last: {robot.last_action}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # calibration overlay
        if calib_mode:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 100, 200), 4)
            cv2.putText(overlay, "CALIBRATE: click the yellow object",
                        (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (0, 200, 255), 2)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # bottom status bar
        cv2.rectangle(frame, (0, h - 52), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, status_text, (8, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "SPACE:speak  C:calibrate  R:reset  Q:quit",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (140, 140, 140), 1, cv2.LINE_AA)

        cv2.imshow("AI Arm Controller", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            stt.listen_async()
            status_text  = "Listening ... speak now"
            status_color = (0, 0, 255)
        elif key == ord("c"):
            calib_mode = True
            print("[CALIB] Click on yellow object in the window")
        elif key == ord("r"):
            robot.reset()
            confirm_count = 0
            status_text   = "Reset → IDLE"
            status_color  = (0, 200, 0)

    cap.release()
    cv2.destroyAllWindows()
    if ser and ser.is_open:
        ser.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
