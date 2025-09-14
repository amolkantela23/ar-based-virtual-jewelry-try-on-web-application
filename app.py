from flask import Flask, render_template, Response, request
import cv2
import numpy as np

app = Flask(__name__)

# Necklace items with details (existing category)
jewelry_items = {
    "Necklace 1": {
        "image": "necklace1.png",
        "name": "Diamond Necklace",
        "description": "Dazzling teardrop diamonds cascade in luxurious elegance.",
        "price": "₹2,00,00,000"
    },
    "Necklace 2": {
        "image": "necklace2.png",
        "name": "Golden Quadrant Necklace",
        "description": "Delicate gold squares create a stylish, modern look.",
        "price": "₹60,000"
    },
    "Necklace 3": {
        "image": "necklace3.png",
        "name": "Entwined Hearts",
        "description": "Graceful heart-shaped loops interlink for timeless elegance.",
        "price": "₹1,50,000"
    },
    "Necklace 4": {
        "image": "necklace4.png",
        "name": "Moonlight Oval Beads",
        "description": "Minimalist silver discs for modern everyday style.",
        "price": "₹10,000"
    },
    "Necklace 5": {
        "image": "necklace5.png",
        "name": "Desert Bloom Dangle",
        "description": "Earthy-toned beads and dangling drops for boho charm.",
        "price": "₹5,000"
    },
    "Necklace 6": {
        "image": "necklace6.png",
        "name": "Tricolor Cascading Beads",
        "description": "Three-tier metallic beads create a graceful drape.",
        "price": "₹80,000"
    },
    "Necklace 7": {
        "image": "necklace7.png",
        "name": "Golden Pearl Heart",
        "description": "Chain links with pearls and a charming heart pendant.",
        "price": "₹40,000"
    },
    "Necklace 8": {
        "image": "necklace8.png",
        "name": "Rose Glow Gem",
        "description": "Soft pink crystals dangle around a shimmering centerpiece.",
        "price": "₹8,000"
    }
}

# Earrings category (new)
earring_items = {
    "Earring 1": {
        "image": "earring1.png",
        "name": "Eclipse Pearl Drop",
        "description": "Contrasting pearls and a sparkling diamond for refined elegance.",
        "price": "₹75,000"
    },
    "Earring 2": {
        "image": "earring2.png",
        "name": "Radiant Teardrop",
        "description": "Sparkling diamonds form an opulent cascading teardrop design.",
        "price": "₹3,00,000"
    },
    "Earring 3": {
        "image": "earring3.png",
        "name": "Midnight Gothic",
        "description": "Intricate black crystals evoke bold, dramatic elegance.",
        "price": "₹25,000"
    },
    "Earring 4": {
        "image": "earring4.png",
        "name": "Royal Ruby Pearl",
        "description": "Glittering rubies and pearls create a regal chandbali design.",
        "price": "₹80,000"
    },
    "Earring 5": {
        "image": "earring5.png",
        "name": "Rose Diamond Teardrop",
        "description": "A graceful rose-gold teardrop with sparkling center diamonds.",
        "price": "₹35,000"
    }
}

# Nose Pins category (new)
nose_pin_items = {
    "Nose Pin 1": {
        "image": "nosepin1.png",
        "name": "Traditional Ruby Pearl",
        "description": "Gold swirl accented by a vivid ruby and pearls.",
        "price": "₹15,000"
    },
    "Nose Pin 2": {
        "image": "nosepin2.png",
        "name": "Oxidized Tribal",
        "description": "Intricate dotted design for a classic boho charm.",
        "price": "₹2000"
    },
    "Nose Pin 3": {
        "image": "nosepin3.png",
        "name": "Elegant Charm",
        "description": "A graceful design that adds a touch of elegance.",
        "price": "₹8000"
    }
}

# Global variables for selected overlays.
# Only one overlay will be active at a time.
selected_jewelry = "Necklace 1"   # default necklace
selected_earring = ""             # none selected by default
selected_nosepin = ""             # none selected by default

# Cache for detection positions
last_face = None      # Format: (x, y, w, h)
last_eyes = None      # List of (x, y, w, h) for up to 2 eyes

# Initialize webcam and lower the resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load cascades for face and eye detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade  = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

def overlay_image(background, overlay, x, y):
    """Overlay a PNG image with alpha channel on the background image at (x, y)."""
    h, w = overlay.shape[:2]
    # Skip overlay if out of bounds
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return
    b, g, r, a = cv2.split(overlay)
    overlay_rgb = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a)) / 255.0
    roi = background[y:y+h, x:x+w]
    blended = (roi * (1 - mask) + overlay_rgb * mask).astype(np.uint8)
    background[y:y+h, x:x+w] = blended

def generate_frames():
    global selected_jewelry, selected_earring, selected_nosepin, last_face, last_eyes
    frame_count = 0  # counter for processing overlays every other frame
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Process overlays on every other frame for performance.
        if frame_count % 2 == 0:
            # If a necklace is selected, process face detection and overlay.
            if selected_jewelry:
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
                )
                if len(faces) > 0:
                    last_face = faces[0]  # use first detected face
                if last_face is not None:
                    (x, y, w, h) = last_face
                    overlay = cv2.imread(f"static/{jewelry_items[selected_jewelry]['image']}", cv2.IMREAD_UNCHANGED)
                    if overlay is not None:
                        orig_h, orig_w = overlay.shape[:2]
                        aspect_ratio = orig_w / orig_h
                        new_width = int(w * 1.0)
                        new_height = int(new_width / aspect_ratio)
                        overlay_resized = cv2.resize(overlay, (new_width, new_height))
                        x_offset = x - int(w * 0.05)
                        y_offset = y + int(h * 0.95)
                        try:
                            overlay_image(frame, overlay_resized, x_offset, y_offset)
                        except Exception as e:
                            print(f"Necklace overlay error: {e}")

            # If an earring is selected, process eye detection and overlay.
            elif selected_earring:
                eyes = eye_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(eyes) > 0:
                    last_eyes = sorted(eyes, key=lambda ex: ex[0])[:2]
                if last_eyes is not None:
                    earring_overlay = cv2.imread(f"static/{earring_items[selected_earring]['image']}", cv2.IMREAD_UNCHANGED)
                    if earring_overlay is not None:
                        for i, (ex, ey, ew, eh) in enumerate(last_eyes):
                            scale_factor = 0.7  # adjust scaling factor for earrings
                            new_width = int(ew * scale_factor)
                            orig_eh, orig_ew = earring_overlay.shape[:2]
                            new_height = int(new_width * (orig_eh / orig_ew))
                            earring_resized = cv2.resize(earring_overlay, (new_width, new_height))
                            if i == 0:
                                x_offset = ex - int(new_width * 0.8)
                            else:
                                x_offset = ex + ew - int(new_width * 0.2)
                            y_offset = ey + eh + 20  # adjust vertical offset
                            try:
                                overlay_image(frame, earring_resized, x_offset, y_offset)
                            except Exception as e:
                                print(f"Earring overlay error: {e}")

            # If a nose pin is selected, process face detection and overlay.
            elif selected_nosepin:
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
                )
                if len(faces) > 0:
                    last_face = faces[0]
                if last_face is not None:
                    (x, y, w, h) = last_face
                    nose_overlay = cv2.imread(f"static/{nose_pin_items[selected_nosepin]['image']}", cv2.IMREAD_UNCHANGED)
                    if nose_overlay is not None:
                        # Scale the nose pin relative to face width.
                        new_width = int(w * 0.3)
                        orig_h, orig_w = nose_overlay.shape[:2]
                        new_height = int(new_width * (orig_h / orig_w))
                        nose_resized = cv2.resize(nose_overlay, (new_width, new_height))
                        # Position the nose pin roughly at the center of the face.
                        x_offset = x + w // 2 + int(new_width * 0.01)
                        y_offset = y + h // 2 - new_height // 7

                        try:
                            overlay_image(frame, nose_resized, x_offset, y_offset)
                        except Exception as e:
                            print(f"Nose pin overlay error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html',
                           jewelry_items=jewelry_items,
                           earring_items=earring_items,
                           nose_pin_items=nose_pin_items)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# When a necklace is selected, clear any other selection.
@app.route('/change_jewelry', methods=['POST'])
def change_jewelry():
    global selected_jewelry, selected_earring, selected_nosepin, last_face
    selected_jewelry = request.form['jewelry']
    selected_earring = ""
    selected_nosepin = ""
    last_face = None  # reset cached face detection
    return "", 204

# When an earring is selected, clear any other selection.
@app.route('/change_earring', methods=['POST'])
def change_earring():
    global selected_earring, selected_jewelry, selected_nosepin, last_eyes
    selected_earring = request.form['earring']
    selected_jewelry = ""
    selected_nosepin = ""
    last_eyes = None  # reset cached eyes detection
    return "", 204

# When a nose pin is selected, clear any other selection.
@app.route('/change_nosepin', methods=['POST'])
def change_nosepin():
    global selected_nosepin, selected_jewelry, selected_earring, last_face
    selected_nosepin = request.form['nosepin']
    selected_jewelry = ""
    selected_earring = ""
    last_face = None  # reset cached face detection
    return "", 204

if __name__ == '__main__':
    app.run(debug=True)
