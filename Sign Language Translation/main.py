from flask import Flask, request, jsonify, render_template, send_file
import os
import cv2
import logging
import shutil
import numpy as np
from mediapipe.python.solutions import drawing_utils, holistic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

UPLOAD_FOLDER = 'uploads'
CONVERTED_FOLDER = 'converted'
KEYPOINTS_FOLDER = 'keypoints'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(CONVERTED_FOLDER):
    os.makedirs(CONVERTED_FOLDER)

if not os.path.exists(KEYPOINTS_FOLDER):
    os.makedirs(KEYPOINTS_FOLDER)

mp_drawing = drawing_utils
mp_holistic = holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def process_keypoints(input_folder, window_size):
    X = []
    y = []
    for root, dirs, files in os.walk(KEYPOINTS_FOLDER):
        total_files = len(files)
        padding_needed = window_size - total_files % window_size
        windows = [files[i:i+window_size] for i in range(0, total_files, window_size)]
        if padding_needed > 0 and padding_needed != window_size:
            windows[-1] = windows[-2][-(padding_needed):] + windows[-1]
        for window_files in windows:
            window_keypoints = []
            for file in window_files:
                if file.endswith(".npy"):
                    filepath = os.path.join(root, file)
                    keypoints = np.load(filepath)
                    window_keypoints.append(keypoints)
            X.append(np.array(window_keypoints))
            # Assuming the folder name is the label
            label = os.path.basename(root)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video file provided."}), 400
    
    video_file = request.files['video']
    video_file_name = video_file.filename
    video_file_path_upload = os.path.join(UPLOAD_FOLDER, video_file_name)
    video_file_path_convert = os.path.join(CONVERTED_FOLDER, video_file_name)
    
    video_file.save(video_file_path_upload)
    shutil.copy(video_file_path_upload, video_file_path_convert)
    
    converted_result, frames_elapsed = convert_video_to_avi(video_file_path_convert, CONVERTED_FOLDER)
    keypoint_path = os.path.join(KEYPOINTS_FOLDER, video_file_name.split('.')[0])
    if os.path.exists(video_file_path_upload):
        return jsonify({"success": True, "filename": video_file_name, 'keypoints_folder': keypoint_path, 'frames_count': frames_elapsed})
    else:
        return jsonify({"success": False, "error": "Failed to save uploaded file."}), 500

@app.route('/play/<filename>')
def play_file(filename):
    return send_file(os.path.join(CONVERTED_FOLDER, filename))

def convert_video_to_avi(video_file_path, converted_folder):
    if not os.path.isfile(video_file_path):
        logging.error("Error: The provided path is not a file.")
        return {"success": False, "error": "The provided path is not a file."}
    
    if not video_file_path.endswith((".mp4", ".MOV", ".mkv", ".mpeg", ".MP4")):
        logging.error("Error: The provided file is not a supported video format.")
        return {"success": False, "error": "The provided file is not a supported video format."}
    
    try:
        video_capture = cv2.VideoCapture(video_file_path)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        out_file_path = os.path.splitext(os.path.basename(video_file_path))[0] + ".avi"
        out_file_path = os.path.join(converted_folder, out_file_path)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_file_path, fourcc, fps, (frame_width, frame_height))
        frames_count = 0
        while True:
            frames_count += 1
            ret, frame = video_capture.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        video_capture.release()
        
        logging.info("Video successfully converted to AVI format.")
        return {"success": True, "output_path": out_file_path}, frames_count
    except Exception as e:
        logging.error("Error: %s", str(e))
        return {"success": False, "error": str(e)}

def generate_keypoints(video_file_path):
    try:
        video_capture = cv2.VideoCapture(video_file_path)
        frame_num = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            # Extract keypoints from each frame and save to npy file
            image, results = mediapipe_detection(frame, mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5))
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(KEYPOINTS_FOLDER, os.path.splitext(os.path.basename(video_file_path))[0], f"frame_{frame_num}.npy")
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, keypoints)
            frame_num += 1
        video_capture.release()
        logging.info("Keypoints successfully extracted.")
        return {"success": True}
    except Exception as e:
        logging.error("Error: %s", str(e))
        return {"success": False, "error": str(e)}

def get_network_more_wider(frames, input_size, num_classes):
    model = Sequential([
        LSTM(1024, input_shape=(frames, input_size), dropout=0.2),
        Dense(num_classes, activation='softmax', name='output1')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@app.route('/process', methods=['POST'])
def process_data():
    # Generate keypoints 
    video_file = request.files['video']
    video_file_name = video_file.filename
    video_file_path_upload = os.path.join(UPLOAD_FOLDER, video_file_name)
    video_file_path_convert = os.path.join(CONVERTED_FOLDER, video_file_name)

    generate_keypoints(video_file_path_convert)
    input_folder = KEYPOINTS_FOLDER
    window_size = 30
    X, y = process_keypoints(input_folder, window_size)
    
    # Load the model
    model = get_network_more_wider(30, 1662, 100)
    model.load_weights('model_files/ws_30_get_network_more_wider.h5')
    
    # Perform predictionsx
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    counts = np.bincount(y_pred_classes)
    most_common_number = int(np.argmax(counts))
    logging.info("Predicted sentence:", str(most_common_number))
    # Print predictions to log
    logging.info("Predicted classes: %s", str(y_pred_classes))
    
    label_mapping = {
    0: 'Do me a favour.',
    1: 'Do not make me angry.',
    2: 'Do not take it to heart.',
    3: 'Do not worry.',
    4: 'Do you need something?',
    5: 'Go and sleep.',
    6: 'Had your food?',
    7: 'He came by train.',
    8: 'He is going into the room.',
    9: 'He is on the way.',
    10: 'He/she is my friend.',
    11: 'He would be coming today.',
    12: 'Help me.',
    13: 'Hi, how are you?',
    14: 'How are things?',
    15: 'How can I help you?',
    16: 'How can I trust you?',
    17: 'How dare you!',
    18: 'How old are you?',
    19: 'I am (age).',
    20: 'I am afraid of that.',
    21: 'I am crying.',
    22: 'I am feeling bored.',
    23: 'I am feeling cold.',
    24: 'I am fine. Thank you, sir.',
    25: 'I am hungry.',
    26: 'I am in a dilemma what to do.',
    27: 'I am not really sure.',
    28: 'I am really grateful.',
    29: 'I am sitting in the class.',
    30: 'I am so sorry to hear that.',
    31: 'I am suffering from fever.',
    32: 'I am tired.',
    33: 'I am very happy.',
    34: 'I cannot help you there.',
    35: 'I do not agree.',
    36: 'I do not like it.',
    37: 'I do not mean it.',
    38: "I don't agree.",
    39: 'I enjoyed a lot.',
    40: 'I got hurt.',
    41: 'I like you, I love you.',
    42: 'I need water.',
    43: 'I promise.',
    44: 'I really appreciate it.',
    45: 'I somehow got to know about it.',
    46: 'I was stopped by someone.',
    47: 'It does not make any difference to me.',
    48: 'It was nice chatting with you.',
    49: 'Let him take time.',
    50: 'My name is xxxxxxxx.',
    51: 'Nice to meet you.',
    52: "No need to worry, don't worry.",
    53: 'Now onwards, he will never hurt you.',
    54: 'Pour some more water into the glass.',
    55: 'Prepare the bed.',
    56: 'Serve the food.',
    57: 'Shall we go outside?',
    58: 'Speak softly.',
    59: 'Take care of yourself.',
    60: 'Tell me the truth.',
    61: 'Thank you so much.',
    62: 'That is so kind of you.',
    63: 'This place is beautiful.',
    64: 'Try to understand.',
    65: 'Turn on the light, turn off the light.',
    66: 'We are all with you.',
    67: 'Wear the shirt.',
    68: 'What are you doing?',
    69: 'What did you tell him?',
    70: 'What do you think?',
    71: 'What do you want to become?',
    72: 'What happened?',
    73: 'What have you planned for your career?',
    74: 'What is your phone number?',
    75: 'What do you want?',
    76: 'When will the train leave?',
    77: 'Where are you from?',
    78: 'Who are you?',
    79: 'Why are you angry?',
    80: 'Why are you crying?',
    81: 'Why are you disappointed?',
    82: 'You are bad.',
    83: 'You are good.',
    84: 'You are welcome.',
    85: 'You can do it.',
    86: 'You do anything, I do not care.',
    87: 'Do not be stubborn.',
    88: 'You need a medicine, take this one.',
    89: 'Which college/school are you from?',
    90: 'Do not hurt me.',
    91: 'Are you free today?',
    92: 'Are you hiding something?',
    93: 'Bring water for me.',
    94: 'Can I help you?',
    95: 'Can you repeat that please?',
    96: 'Comb your hair.',
    97: 'Congratulations!',
    98: 'Could you please talk slower?',
    99: 'Do not abuse him.'
    }

    # Get the string corresponding to the most common number
    most_common_string = label_mapping[most_common_number]

    logging.info("Most common string:", most_common_string)

    shutil.rmtree(KEYPOINTS_FOLDER)
    # shutil.rmtree(CONVERTED_FOLDER)
    # shutil.rmtree(UPLOAD_FOLDER)
    return jsonify({"success": True, "prediction": most_common_string})

if __name__ == '__main__':
    app.run(debug=True)
