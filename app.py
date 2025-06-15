import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from solve_captcha import solve_captcha

# Yüklenen dosyaların kaydedileceği klasör ve izin verilen uzantılar
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    # Dosya uzantısı kontrolü
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ana sayfa: Görsel yükleme ve tahmin sonucu gösterme
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None
    error = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'Dosya bulunamadı!'
        else:
            file = request.files['file']
            if not file or not file.filename:
                error = 'Dosya seçilmedi!'
            elif not allowed_file(file.filename):
                error = 'Geçersiz dosya formatı! Sadece PNG, JPG ve JPEG dosyaları yükleyebilirsiniz.'
            else:
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    prediction = solve_captcha(filepath)
                    image_url = url_for('uploaded_file', filename=filename)
                except Exception as e:
                    error = f'Tahmin sırasında hata oluştu: {str(e)}'
    
    return render_template('index.html', prediction=prediction, image_url=image_url, error=error)

# Yüklenen görseli sunmak için
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Flask uygulamasını başlat
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 