import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
from train import CaptchaModel, IMG_HEIGHT, IMG_WIDTH, NUM_DIGITS

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CaptchaModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, device

def predict_captcha(model, device, image_path):
    # Görüntü dönüşümleri
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Görüntüyü yükle ve işle
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)
    
    # Tahmin yap
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(2)
        predicted = predicted.squeeze().cpu().numpy()
        
    # Tahmini string'e çevir
    prediction = ''.join(map(str, predicted))
    return prediction

def main():
    # Modeli yükle
    model, device = load_model('captcha_model.pth')
    print(f"Model yüklendi. Kullanılan cihaz: {device}")
    
    # Test görüntülerini tahmin et
    test_results = []
    
    # Test görüntülerini oku
    with open('captcha_data.csv', 'r') as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            if 'test' in row['image_path']:  # Sadece test görüntülerini al
                image_path = row['image_path']
                true_label = str(row['solution']).zfill(NUM_DIGITS)
                
                # Tahmin yap
                prediction = predict_captcha(model, device, image_path)
                
                # Sonucu kaydet
                test_results.append({
                    'image_path': image_path,
                    'predicted': prediction,
                    'true': true_label,
                    'correct': prediction == true_label
                })
                
                # Sonucu ekrana yazdır
                print(f"\nGörüntü: {image_path}")
                print(f"Tahmin: {prediction}")
                print(f"Gerçek: {true_label}")
                print(f"Doğru mu: {'✅' if prediction == true_label else '❌'}")
    
    # Sonuçları kaydet
    with open('test_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)
    
    # Genel doğruluk oranını hesapla
    correct = sum(1 for r in test_results if r['correct'])
    total = len(test_results)
    accuracy = 100. * correct / total
    
    print(f"\nTest Sonuçları:")
    print(f"Toplam görüntü: {total}")
    print(f"Doğru tahmin: {correct}")
    print(f"Doğruluk oranı: {accuracy:.2f}%")

if __name__ == '__main__':
    main() 