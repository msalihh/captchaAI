import torch
from torchvision import transforms
from PIL import Image
from train import CaptchaModel, IMG_HEIGHT, IMG_WIDTH

# Tek bir görsel için CAPTCHA tahmini yapan fonksiyon
def solve_captcha(image_path):
    # GPU/CPU kontrolü
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Eğitilmiş modeli yükle
    model = CaptchaModel().to(device)
    checkpoint = torch.load('captcha_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Görüntü dönüşümleri (boyutlandırma, normalize)
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Train.py ile aynı normalize değerleri
    ])
    
    # Görüntüyü yükle ve işle
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Model ile tahmin yap
    with torch.no_grad():
        outputs = model(image_tensor)
        preds = outputs.argmax(dim=2)
        predicted = preds.squeeze().cpu().numpy()
    
    # Tahmini string olarak döndür
    return ''.join(map(str, predicted))

# Komut satırından görsel yolu alıp tahmin yapan ana blok
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("Kullanım: python solve_captcha.py <görüntü_yolu>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    try:
        result = solve_captcha(image_path)
        print(f"CAPTCHA Çözümü: {result}")
    except Exception as e:
        print(f"Hata: {e}") 