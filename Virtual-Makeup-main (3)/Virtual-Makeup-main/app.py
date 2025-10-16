from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
import os
import socket
import base64
from io import BytesIO
from PIL import Image
from utils import apply_foundation, apply_foundation_to_uploaded_image
from config import app_state

app = Flask(__name__)

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create a single camera instance
camera_instance = None

class VideoCamera:
    def __init__(self):
        self.video = None
        # Try different camera indices
        for camera_index in [0, 1, 2, 3]:
            temp_video = cv2.VideoCapture(camera_index)
            if temp_video.isOpened():
                self.video = temp_video
                print(f"✅ Camera found at index {camera_index}")
                break
            else:
                print(f"❌ Camera not found at index {camera_index}")
        
        if self.video and self.video.isOpened():
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video.set(cv2.CAP_PROP_FPS, 30)
        else:
            print("❌ No camera found - running in demo mode")
    
    
    def get_frame(self):
        if not self.video or not self.video.isOpened():
            return self.get_demo_frame()
            
        success, frame = self.video.read()
        if not success:
            return self.get_demo_frame()
            
        frame = cv2.flip(frame, 1)
        
        # Apply foundation
        frame = apply_foundation(
            frame, 
            shade=app_state.current_shade,
            intensity=app_state.current_intensity
        )
        
        # Encode frame
        ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpeg.tobytes() if ret else self.get_demo_frame()
    
    def get_demo_frame(self):
        """Generate a demo frame when camera is not available"""
        demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create a nice demo display
        cv2.putText(demo_frame, "VIRTUAL FOUNDATION TRY-ON", (80, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(demo_frame, "Camera is working on local network", (60, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
        
        # Show current shade
        shade_info = app_state.get_current_shade_info()
        cv2.putText(demo_frame, f"Current Shade: {shade_info['name']}", (180, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(demo_frame, f"Undertone: {shade_info['undertone']}", (200, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        
        # Instructions
        cv2.putText(demo_frame, "Use the controls on the right to change shades", (50, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(demo_frame, "Camera will work when accessed via network URL", (40, 380), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        ret, jpeg = cv2.imencode('.jpg', demo_frame)
        return jpeg.tobytes() if ret else None

def get_camera():
    """Get or create the camera instance"""
    global camera_instance
    if camera_instance is None:
        camera_instance = VideoCamera()
    return camera_instance

def gen_frames():
    """Generate frames using the single camera instance"""
    camera = get_camera()
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# API Routes
@app.route('/api/state')
def get_state():
    return jsonify(app_state.get_state())

@app.route('/api/shades')
def get_shades():
    return jsonify({
        'shades': app_state.get_available_shades(),
        'current': app_state.current_shade
    })

@app.route('/api/set_shade', methods=['POST'])
def set_shade():
    data = request.get_json()
    shade = data.get('shade', '03')
    
    if app_state.set_shade(shade):
        return jsonify({'status': 'success', 'shade': shade})
    return jsonify({'status': 'error', 'message': 'Invalid shade'})

@app.route('/api/set_intensity', methods=['POST'])
def set_intensity():
    data = request.get_json()
    intensity = float(data.get('intensity', 0.7))
    
    if app_state.set_intensity(intensity):
        return jsonify({'status': 'success', 'intensity': intensity})
    return jsonify({'status': 'error', 'message': 'Invalid intensity'})

@app.route('/api/toggle_foundation', methods=['POST'])
def toggle_foundation():
    data = request.get_json()
    enabled = data.get('enabled', True)
    app_state.foundation_enabled = enabled
    return jsonify({'status': 'success', 'enabled': enabled})

@app.route('/api/reset')
def reset_foundation():
    app_state.reset()
    return jsonify({'status': 'success', 'message': 'Foundation reset'})

@app.route('/api/network_info')
def network_info():
    """Get network information for easy access"""
    local_ip = get_local_ip()
    return jsonify({
        'local_ip': local_ip,
        'port': 5000,
        'local_url': f'http://localhost:5000',
        'network_url': f'http://{local_ip}:5000'
    })

# New routes for image upload and processing
@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image file provided'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'})
        
        # Read and process the image
        file_data = file.read()
        nparr = np.frombuffer(file_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'status': 'error', 'message': 'Invalid image file'})
        
        print(f"📸 Processing uploaded image: {img.shape[1]}x{img.shape[0]}")
        
        # Save original image for comparison
        original_filename = f"original_{os.urandom(4).hex()}.jpg"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        cv2.imwrite(original_path, img)
        
        # Apply foundation to the uploaded image using the special function
        processed_img = apply_foundation_to_uploaded_image(
            img, 
            shade=app_state.current_shade,
            intensity=app_state.current_intensity
        )
        
        # Check if foundation was applied (compare with original)
        if np.array_equal(processed_img, img):
            print("⚠️ No foundation applied - no face detected or foundation disabled")
            return jsonify({
                'status': 'error', 
                'message': 'No face detected in the image. Please upload a clear photo with a visible face.'
            })
        
        # Save processed image
        filename = f"processed_{os.urandom(4).hex()}.jpg"
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        cv2.imwrite(processed_path, processed_img)
        
        print(f"✅ Foundation applied successfully to uploaded image")
        
        return jsonify({
            'status': 'success', 
            'original_image': original_filename,
            'processed_image': filename,
            'message': 'Foundation applied successfully'
        })
        
    except Exception as e:
        print(f"❌ Error processing uploaded image: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Error processing image: {str(e)}'})

@app.route('/api/process_base64', methods=['POST'])
def process_base64():
    """Process base64 encoded image"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'message': 'No image data provided'})
        
        # Extract base64 data
        image_data = data['image']
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'status': 'error', 'message': 'Invalid image data'})
        
        print(f"📸 Processing base64 image: {img.shape[1]}x{img.shape[0]}")
        
        # Apply foundation using the special function for uploaded images
        processed_img = apply_foundation_to_uploaded_image(
            img, 
            shade=app_state.current_shade,
            intensity=app_state.current_intensity
        )
        
        # Check if foundation was applied
        if np.array_equal(processed_img, img):
            print("⚠️ No foundation applied - no face detected")
            return jsonify({
                'status': 'error', 
                'message': 'No face detected in the image. Please try a different photo.'
            })
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print(f"✅ Foundation applied successfully to base64 image")
        
        return jsonify({
            'status': 'success',
            'processed_image': f'data:image/jpeg;base64,{processed_base64}'
        })
        
    except Exception as e:
        print(f"❌ Error processing base64 image: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Error processing image: {str(e)}'})

@app.route('/uploads/<filename>')
def get_uploaded_image(filename):
    """Serve original uploaded images"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/processed/<filename>')
def get_processed_image(filename):
    """Serve processed images"""
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))

def get_local_ip():
    """Get the local IP address for network access"""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    port = 5000
    local_ip = get_local_ip()
    
    print("🎨" * 50)
    print("🎨 VIRTUAL FOUNDATION TRY-ON - NETWORK EDITION")
    print("🎨" * 50)
    print()
    print("📍 ACCESS URLs:")
    print(f"   💻 Local:  http://localhost:{port}")
    print(f"   🌐 Network: http://{local_ip}:{port}")
    print()
    
    # Initialize camera once at startup
    camera = get_camera()
    if camera.video and camera.video.isOpened():
        print("📸 CAMERA: ✅ Connected and Ready!")
    else:
        print("📸 CAMERA: ❌ Not found (showing demo mode)")
    print()
    
    print("📸 UPLOAD FEATURE: ✅ Ready to process uploaded images")
    print("🎨 FOUNDATION: ✅ Ready for both live stream and uploaded photos")
    print()
    
    print("👔 SHARING INSTRUCTIONS:")
    print(f"   1. Share this URL with your boss: http://{local_ip}:{port}")
    print("   2. Make sure you're on the same Wi-Fi network")
    print("   3. Your boss opens the URL in any browser")
    print("   4. They allow camera access when prompted")
    print()
    print("🛡️  SECURITY: Only people on your local network can access")
    print("⏹️  PRESS Ctrl+C to stop the application")
    print("-" * 50)
    
    # Run on all network interfaces
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)