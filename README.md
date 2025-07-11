# 6 Haneli CAPTCHA Çözücü

Bu proje, 6 haneli sayısal CAPTCHA görsellerini çözmek için PyTorch tabanlı bir derin öğrenme modeli içerir.

## Özellikler

- ResNet18 tabanlı transfer learning
- Mixed precision training (AMP)
- Early stopping
- Cosine learning rate scheduling
- GPU desteği
- Detaylı eğitim metrikleri
- Flask tabanlı web arayüzü

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/kullaniciadi/captcha_ai.git
cd captcha_ai
```

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv venv
# Windows için:
venv\Scripts\activate
# Linux/Mac için:
source venv/bin/activate
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. Veri setini hazırlayın:
- `train-images/` klasörüne eğitim görsellerini yerleştirin
- `validation-images/` klasörüne doğrulama görsellerini yerleştirin
- `captcha_data.csv` dosyasını oluşturun (image_path ve solution sütunları)

## Kullanım

### Model Eğitimi

Modeli eğitmek için:
```bash
python train.py
```

### Tahmin Yapma

Tek bir görsel için tahmin yapmak için:
```bash
python predict.py --image path/to/image.png
```

### Web Arayüzü

Web arayüzünü başlatmak için:
```bash
python app.py
```
Tarayıcınızda `http://localhost:5000` adresine gidin.

## Proje Yapısı

```
captcha_ai/
├── app.py                 # Flask web uygulaması
├── train.py              # Model eğitim kodu
├── predict.py            # Tahmin kodu
├── solve_captcha.py      # CAPTCHA çözme yardımcı fonksiyonları
├── requirements.txt      # Bağımlılıklar
└── README.md            # Bu dosya
```

## Model Mimarisi

- Backbone: ResNet18 (transfer learning)
- Giriş: 1x50x200 gri tonlamalı görüntü
- Çıkış: 6 haneli sayı tahmini (her hane için 0-9 arası)

## Lisans

MIT License

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: X'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun #   c a p t c h a A I  
 