from flask import Flask, request, render_template, send_from_directory
from flask_cors import CORS  # CORS enable karne ke liye
import os
from realesrgan.realesrganer import RealESRGANer  # realesrgan folder se import
from basicsr.archs.rrdbnet_arch import RRDBNet  # basicsr folder se import
import cv2

app = Flask(__name__)
CORS(app)  # CORS enable karein
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Uploaded images ka folder
app.config['ENHANCED_FOLDER'] = 'static/enhanced_images'  # Enhanced images ka folder

# Real-ESRGAN model load karein
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4
model_path = 'weights/RealESRGAN_x4plus.pth'  # Weights file ka path
upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        if file:
            # Uploaded image ko save karein
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(upload_path)

            # Image ko enhance karein
            img = cv2.imread(upload_path, cv2.IMREAD_UNCHANGED)
            output, _ = upsampler.enhance(img, outscale=4)

            # Enhanced image ko save karein
            enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], f"enhanced_{file.filename}")
            cv2.imwrite(enhanced_path, output)

            return render_template('index.html', original_image=upload_path, enhanced_image=enhanced_path)
    return render_template('index.html')

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['ENHANCED_FOLDER'], filename)

if __name__ == '__main__':
    # Ensure upload and enhanced folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['ENHANCED_FOLDER'], exist_ok=True)
    app.run(debug=True)
