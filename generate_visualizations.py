import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re
import logging
import matplotlib.patches as mpatches
import json
import cv2
import torch
from torchsummary import summary
from train import CaptchaModel
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Log dosyasından doğruluk verisi çıkarma fonksiyonu
def extract_accuracy_data(log_file):
    epochs = []
    train_accs = []
    val_accs = []
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'.*Epoch (\d+): Train Loss: [\d.]+,\s*Acc: ([\d.]+)%,\s*Val Loss: [\d.]+,\s*Acc: ([\d.]+)%', line)
            if match:
                epoch = int(match.group(1))
                train_acc = float(match.group(2))
                val_acc = float(match.group(3))
                epochs.append(epoch)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
    return epochs, train_accs, val_accs

# 1. Veri Seti Örnekleri
def create_dataset_samples():
    df = pd.read_csv('captcha_data.csv')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CAPTCHA Veri Seti Örnekleri', fontsize=16)
    
    for idx, (_, row) in enumerate(df.head(6).iterrows()):
        img_path = row['image_path']
        solution = row['solution']
        
        img = Image.open(img_path).convert('L')
        ax = axes[idx // 3, idx % 3]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Etiket: {solution}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Model Mimarisi Diyagramı
def create_model_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Katman bloklarını çiz
    layers = [
        ('Input\n(1, 50, 200)', 0),
        ('Conv2D + ReLU\n(32, 25, 100)', 1),
        ('MaxPool2D\n(32, 12, 50)', 2),
        ('Conv2D + ReLU\n(64, 6, 25)', 3),
        ('MaxPool2D\n(64, 3, 12)', 4),
        ('Flatten\n(2304)', 5),
        ('FC Layers\n(6 x 512 → 10)', 6)
    ]
    
    # Blokları çiz
    for i, (name, pos) in enumerate(layers):
        rect = mpatches.Rectangle((pos, 0), 1, 1, facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(pos + 0.5, 0.5, name, ha='center', va='center')
        
        if i < len(layers) - 1:
            ax.arrow(pos + 1, 0.5, 0.2, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')
    plt.title('CNN Model Mimarisi', fontsize=16)
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Eğitim Sonuçları Grafiği (Sadece doğruluk)
def create_accuracy_plot():
    epochs, train_accs, val_accs = extract_accuracy_data('training.log')
    if not epochs:
        print('Doğruluk verileri çıkarılamadı!')
        return
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label='Eğitim Doğruluğu', marker='o')
    plt.plot(epochs, val_accs, label='Doğrulama Doğruluğu', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk (%)')
    plt.title('Eğitim ve Doğrulama Doğrulukları')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_training_history(log_file, output_file):
    epochs = []
    train_losses = []
    val_losses = []
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'.*Epoch (\d+): Train Loss: ([\d.]+),\s*Acc: [\d.]+%,\s*Val Loss: ([\d.]+),\s*Acc: [\d.]+%', line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    with open(output_file, 'w') as f:
        json.dump(history, f)

# --- 1. İleri Besleme Blok Diyagramı ---
def draw_block_diagram():
    fig, ax = plt.subplots(figsize=(14, 3))
    layers = [
        ("Input\n(1,50,200)", 0),
        ("Conv2D\n(32,25,100)", 1),
        ("ReLU", 2),
        ("MaxPool\n(32,12,50)", 3),
        ("Conv2D\n(64,12,50)", 4),
        ("ReLU", 5),
        ("MaxPool\n(64,6,25)", 6),
        ("Conv2D\n(128,6,25)", 7),
        ("ReLU", 8),
        ("MaxPool\n(128,3,12)", 9),
        ("Conv2D\n(256,3,12)", 10),
        ("ReLU", 11),
        ("MaxPool\n(256,1,6)", 12),
        ("Flatten\n(1536)", 13),
        ("FC\n(512)", 14),
        ("Output\n(6x10)", 15)
    ]
    for i, (name, pos) in enumerate(layers):
        rect = mpatches.FancyBboxPatch((pos, 0), 0.9, 1, boxstyle="round,pad=0.1", facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(pos+0.45, 0.5, name, ha='center', va='center', fontsize=10)
        if i < len(layers)-1:
            ax.arrow(pos+0.9, 0.5, 0.1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black', length_includes_head=True)
    ax.set_xlim(-0.5, len(layers)-0.1)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')
    plt.title('İleri Besleme Blok Diyagramı')
    plt.savefig('block_diagram.png', bbox_inches='tight', dpi=300)
    plt.close()

# --- 2. 3x3 Kernel ile Convolution İşlemi ---
def draw_conv_example():
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    input_patch = np.array([[1,2,3,0,1],[0,1,2,3,1],[1,0,2,1,2],[2,1,0,1,0],[1,2,1,0,1]])
    kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    bias = 1
    # Çıktı hesaplama
    output = np.zeros((3,3), dtype=int)
    for i in range(3):
        for j in range(3):
            region = input_patch[i:i+3, j:j+3]
            output[i,j] = np.sum(region * kernel) + bias
    # Input
    axs[0].imshow(input_patch, cmap='Blues', vmin=0)
    axs[0].set_title('Input Patch')
    for (i, j), val in np.ndenumerate(input_patch):
        axs[0].text(j, i, str(val), ha='center', va='center', fontsize=10)
    axs[0].axis('off')
    # Kernel
    axs[1].imshow(kernel, cmap='coolwarm', vmin=-1, vmax=1)
    axs[1].set_title('3x3 Kernel\nBias=1')
    for (i, j), val in np.ndenumerate(kernel):
        axs[1].text(j, i, str(val), ha='center', va='center', fontsize=10)
    axs[1].axis('off')
    # Output
    axs[2].imshow(output, cmap='Greens')
    axs[2].set_title('Output Feature Map')
    for (i, j), val in np.ndenumerate(output):
        axs[2].text(j, i, str(val), ha='center', va='center', fontsize=10)
    axs[2].axis('off')
    plt.suptitle('3x3 Kernel ile Convolution İşlemi')
    plt.savefig('conv_example.png', bbox_inches='tight', dpi=300)
    plt.close()

# --- 3. 2x2 MaxPooling Örneği ---
def draw_maxpool_example():
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    input_mat = np.array([[1,3,2,4],[5,6,1,2],[0,2,3,1],[4,1,2,0]])
    output_mat = np.array([[6,4],[4,3]])
    # Input
    axs[0].imshow(input_mat, cmap='Blues')
    axs[0].set_title('Input (4x4)')
    for (i, j), val in np.ndenumerate(input_mat):
        axs[0].text(j, i, str(val), ha='center', va='center', fontsize=10)
    axs[0].axis('off')
    # Output
    axs[1].imshow(output_mat, cmap='Greens')
    axs[1].set_title('Output (2x2)')
    for (i, j), val in np.ndenumerate(output_mat):
        axs[1].text(j, i, str(val), ha='center', va='center', fontsize=10)
    axs[1].axis('off')
    plt.suptitle('2x2 MaxPooling Örneği')
    plt.savefig('maxpool_example.png', bbox_inches='tight', dpi=300)
    plt.close()

# --- 4. Classifier Şeması ---
def draw_classifier_schema():
    fig, ax = plt.subplots(figsize=(8, 3))
    # Flatten
    ax.add_patch(mpatches.FancyBboxPatch((0, 0.4), 1, 0.6, boxstyle="round,pad=0.1", facecolor='#e0e0e0', edgecolor='black'))
    ax.text(0.5, 0.7, 'Flatten\n(1536)', ha='center', va='center', fontsize=10)
    # FC
    ax.add_patch(mpatches.FancyBboxPatch((1.5, 0.4), 1, 0.6, boxstyle="round,pad=0.1", facecolor='#b3cde0', edgecolor='black'))
    ax.text(2, 0.7, 'FC\n(512)', ha='center', va='center', fontsize=10)
    # 6 Output
    for i in range(6):
        ax.add_patch(mpatches.FancyBboxPatch((3.2, 0.2 + i*0.12), 1, 0.18, boxstyle="round,pad=0.1", facecolor='#fbb4ae', edgecolor='black'))
        ax.text(3.7, 0.29 + i*0.12, f'Output {i+1}\n(10 sınıf)', ha='center', va='center', fontsize=9)
        ax.arrow(2.5, 0.7, 0.7, 0.12*(i-2.5), head_width=0.04, head_length=0.08, fc='black', ec='black', length_includes_head=True)
    # Oklar
    ax.arrow(1, 0.7, 0.5, 0, head_width=0.08, head_length=0.08, fc='black', ec='black', length_includes_head=True)
    ax.arrow(0, 0.7, 0.5, 0, head_width=0.08, head_length=0.08, fc='black', ec='black', length_includes_head=True)
    ax.set_xlim(-0.2, 4.5)
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    plt.title('Classifier Şeması')
    plt.savefig('classifier_schema.png', bbox_inches='tight', dpi=300)
    plt.close()

# --- Tüm görselleri oluştur ---
def create_all_custom_diagrams():
    draw_block_diagram()
    draw_conv_example()
    draw_maxpool_example()
    draw_classifier_schema()

def print_model_summary():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CaptchaModel().to(device)
    checkpoint = torch.load('captcha_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    summary(model, (1, 50, 200))

# --- Görsel 1: CNN İleri ve Geri Yayılım Diyagramı ---
def draw_forward_backward_diagram():
    fig, ax = plt.subplots(figsize=(12, 3))
    layers = [
        ("Input", 0),
        ("Conv2D", 1),
        ("ReLU", 2),
        ("MaxPool", 3),
        ("Flatten", 4),
        ("FC", 5),
        ("Output", 6)
    ]
    # Bloklar
    for i, (name, pos) in enumerate(layers):
        rect = mpatches.FancyBboxPatch((pos, 0.5), 0.9, 0.8, boxstyle="round,pad=0.1", facecolor='#b3cde0', edgecolor='black')
        ax.add_patch(rect)
        ax.text(pos+0.45, 0.9, name, ha='center', va='center', fontsize=11)
    # Forward okları
    for i in range(len(layers)-1):
        ax.arrow(i+0.9, 0.9, 0.1, 0, head_width=0.08, head_length=0.08, fc='green', ec='green', length_includes_head=True)
    ax.text(3, 1.15, 'Forward', color='green', fontsize=12, fontweight='bold')
    # Backward okları
    for i in range(len(layers)-1, 0, -1):
        ax.arrow(i, 0.7, -0.8, 0, head_width=0.08, head_length=0.08, fc='red', ec='red', length_includes_head=True)
    ax.text(3, 0.55, 'Backward', color='red', fontsize=12, fontweight='bold')
    # Yan açıklamalar
    ax.text(7.2, 1.05, 'Loss hesaplanır', fontsize=10, color='black')
    ax.text(7.2, 0.9, 'Gradients hesaplanır', fontsize=10, color='black')
    ax.text(7.2, 0.75, 'Ağırlıklar güncellenir', fontsize=10, color='black')
    ax.set_xlim(-0.2, 8.5)
    ax.set_ylim(0.4, 1.3)
    ax.axis('off')
    plt.title('CNN İleri ve Geri Yayılım Diyagramı')
    plt.savefig('cnn_forward_backward.png', bbox_inches='tight', dpi=300)
    plt.close()

# --- Görsel 2: Confusion Matrix ve Metrik Formülleri ---
def draw_confusion_matrix_diagram():
    fig, ax = plt.subplots(figsize=(6, 5))
    # 2x2 matris
    matrix = np.array([["TP", "FP"], ["FN", "TN"]])
    ax.imshow([[1,2],[3,4]], cmap='Pastel1', vmin=1, vmax=4)
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, val, ha='center', va='center', fontsize=18, fontweight='bold', color='black')
    # Oklar ve açıklamalar
    ax.text(-0.5, 0, 'Gerçek Pozitif', fontsize=10, color='black')
    ax.text(-0.5, 1, 'Gerçek Negatif', fontsize=10, color='black')
    ax.text(0, -0.3, 'Tahmin Pozitif', fontsize=10, color='black')
    ax.text(1, -0.3, 'Tahmin Negatif', fontsize=10, color='black')
    # Formüller
    ax.text(2.5, 0.2, 'Precision = TP / (TP + FP)', fontsize=11, color='navy')
    ax.text(2.5, 0.5, 'Recall = TP / (TP + FN)', fontsize=11, color='navy')
    ax.text(2.5, 0.8, 'F1 = 2 * (Prec * Rec) / (Prec + Rec)', fontsize=11, color='navy')
    ax.axis('off')
    plt.title('Confusion Matrix ve Metrik Formülleri')
    plt.savefig('confusion_matrix_diagram.png', bbox_inches='tight', dpi=300)
    plt.close()

# --- Tüm yeni görselleri oluştur ---
def create_additional_diagrams():
    draw_forward_backward_diagram()
    draw_confusion_matrix_diagram()

def draw_crossentropy_loss_diagram():
    fig, ax = plt.subplots(figsize=(8, 4))

    # Model output (softmax)
    output_probs = [0.05, 0.1, 0.05, 0.7, 0.05, 0.05]
    classes = ['0', '1', '2', '3', '4', '5']
    y_pos = np.arange(len(classes))

    # Bar plot: model output
    bars = ax.barh(y_pos, output_probs, color='#b3cde0', edgecolor='black')
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{output_probs[i]:.2f}", va='center', fontsize=10)

    # Gerçek etiket (ör: class 3)
    ax.barh(3, 1, color='#fbb4ae', alpha=0.3, edgecolor='red', height=0.8)
    ax.text(1.05, 3, "Gerçek Etiket (3)", va='center', fontsize=12, color='red', fontweight='bold')

    # Ok ve açıklama
    ax.annotate('CrossEntropyLoss Hesaplanır',
                xy=(output_probs[3], 3), xytext=(0.5, 5.2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                fontsize=12, ha='center', color='black', fontweight='bold')

    # Alt açıklama
    ax.text(0.5, -1, "Loss = -log(softmax[Gerçek Sınıf])", fontsize=12, ha='center', color='navy')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Softmax Çıkışı (Olasılık)")
    ax.set_title("Model Çıktısı, Gerçek Etiket ve CrossEntropyLoss Hesaplaması")
    ax.set_xlim(0, 1.2)
    ax.set_ylim(-0.5, len(classes)-0.5)
    ax.axis('on')
    plt.tight_layout()
    plt.savefig('crossentropy_loss_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_chain_rule_diagram():
    fig, ax = plt.subplots(figsize=(10, 3))
    # Katman isimleri ve pozisyonları
    layers = [
        ("w", 0.5),
        ("z = w·x + b", 2),
        ("a = f(z)", 4),
        ("L", 6)
    ]
    # Bloklar
    for name, pos in layers:
        rect = mpatches.FancyBboxPatch((pos, 0.7), 1.2, 0.8, boxstyle="round,pad=0.1", facecolor='#b3cde0', edgecolor='black')
        ax.add_patch(rect)
        ax.text(pos+0.6, 1.1, name, ha='center', va='center', fontsize=13, fontweight='bold')
    # Oklar ve zincir kuralı etiketleri
    ax.arrow(1.7, 1.1, 0.3, 0, head_width=0.08, head_length=0.15, fc='black', ec='black', length_includes_head=True)
    ax.text(2.05, 1.22, r'$\frac{\partial z}{\partial w}$', fontsize=13, color='navy', ha='center')
    ax.arrow(3.2, 1.1, 0.3, 0, head_width=0.08, head_length=0.15, fc='black', ec='black', length_includes_head=True)
    ax.text(3.55, 1.22, r'$\frac{\partial a}{\partial z}$', fontsize=13, color='navy', ha='center')
    ax.arrow(5.2, 1.1, 0.3, 0, head_width=0.08, head_length=0.15, fc='black', ec='black', length_includes_head=True)
    ax.text(5.55, 1.22, r'$\frac{\partial L}{\partial a}$', fontsize=13, color='navy', ha='center')
    # Sonuç gradyan
    ax.arrow(6.6, 0.7, -5.7, -0.4, head_width=0.08, head_length=0.15, fc='crimson', ec='crimson', length_includes_head=True)
    ax.text(1.5, 0.35, r'$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$', fontsize=14, color='crimson', ha='left', fontweight='bold')
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 2)
    ax.axis('off')
    plt.title('Zincir Kuralı ile Gradyan Hesabı (Backpropagation)')
    plt.savefig('chain_rule_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_gradient_descent_diagram():
    # Yüzey (parabola)
    x = np.linspace(-3, 3, 400)
    y = x**2

    # Gradient descent adımları
    steps = [-2.5, -1.5, -0.7, -0.2, 0]
    step_points = [(xi, xi**2) for xi in steps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color='royalblue', linewidth=2, label='Kayıp Yüzeyi (Loss Surface)')

    # Adım noktaları ve oklar
    for i, (xi, yi) in enumerate(step_points):
        ax.plot(xi, yi, 'o', color='orange', markersize=10)
        if i < len(step_points)-1:
            x_next, y_next = step_points[i+1]
            ax.arrow(xi, yi, x_next-xi, y_next-yi, head_width=0.12, head_length=0.18, fc='crimson', ec='crimson', length_includes_head=True)
            ax.text((xi+x_next)/2, (yi+y_next)/2+0.5, f"Öğrenme Adımı", color='crimson', fontsize=9, ha='center')

    # Gradyan yönü
    ax.arrow(-1.5, (-1.5)**2, 0.7, -2.5, head_width=0.12, head_length=0.18, fc='green', ec='green', length_includes_head=True)
    ax.text(-1.1, 1.5, "Gradyan Yönü", color='green', fontsize=10, ha='left')

    # Minimum nokta
    ax.plot(0, 0, 'o', color='red', markersize=12)
    ax.text(0, -1.2, "Minimum (Global Min)", color='red', fontsize=11, ha='center', fontweight='bold')

    ax.set_xlabel("Ağırlık (w)")
    ax.set_ylabel("Kayıp (Loss)")
    ax.set_title("Gradient Descent ile Minimuma İlerleme")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('gradient_descent_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_metrics_box():
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')
    text = (
        r"$\mathrm{Doğruluk\ (Accuracy)} = \frac{TP + TN}{TP + TN + FP + FN}$" + "\n\n"
        r"$\mathrm{Kesinlik\ (Precision)} = \frac{TP}{TP + FP}$" + "\n\n"
        r"$\mathrm{Duyarlılık\ (Recall)} = \frac{TP}{TP + FN}$" + "\n\n"
        r"$\mathrm{F1\text{-}Skor} = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$"
    )
    ax.text(0.5, 0.5, text, fontsize=16, ha='center', va='center', bbox=dict(facecolor='#e0e0e0', edgecolor='navy', boxstyle='round,pad=0.7'))
    plt.title("Sınıflandırma Metrikleri Formülleri", fontsize=14)
    plt.savefig('metrics_box.png', dpi=300, bbox_inches='tight')
    plt.close()

def draw_cnn_architecture_block():
    fig, ax = plt.subplots(figsize=(12, 2.5))
    layers = [
        ("Input\n(1,50,200)", 0),
        ("Conv2D\n(32)", 1),
        ("ReLU", 2),
        ("MaxPool", 3),
        ("Conv2D\n(64)", 4),
        ("ReLU", 5),
        ("MaxPool", 6),
        ("Conv2D\n(128)", 7),
        ("ReLU", 8),
        ("MaxPool", 9),
        ("Conv2D\n(256)", 10),
        ("ReLU", 11),
        ("MaxPool", 12),
        ("Flatten", 13),
        ("FC", 14),
        ("Output\n(6x10)", 15)
    ]
    for i, (name, pos) in enumerate(layers):
        rect = mpatches.FancyBboxPatch((pos, 0.5), 0.9, 1, boxstyle="round,pad=0.1", facecolor='#b3cde0', edgecolor='black')
        ax.add_patch(rect)
        ax.text(pos+0.45, 1, name, ha='center', va='center', fontsize=10)
        if i < len(layers)-1:
            ax.arrow(pos+0.9, 1, 0.1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black', length_includes_head=True)
    ax.set_xlim(-0.5, len(layers)-0.1)
    ax.set_ylim(0.4, 1.6)
    ax.axis('off')
    plt.title('CNN Mimari Blok Diyagramı')
    plt.savefig('cnn_architecture_block.png', bbox_inches='tight', dpi=300)
    plt.close()

def draw_train_val_accuracy():
    with open("training_history.json", "r") as f:
        history = json.load(f)
    if "train_acc" in history and "val_acc" in history:
        plt.figure(figsize=(8,5))
        plt.plot(history["train_acc"], label="Eğitim Doğruluğu", marker="o")
        plt.plot(history["val_acc"], label="Doğrulama Doğruluğu", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Doğruluk (%)")
        plt.title("Eğitim ve Doğrulama Doğrulukları")
        plt.legend()
        plt.grid(True)
        plt.savefig("train_val_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close()

def draw_train_val_loss():
    with open("training_history.json", "r") as f:
        history = json.load(f)
    plt.figure(figsize=(8,5))
    plt.plot(history["train_loss"], label="Eğitim Kaybı", marker="o")
    plt.plot(history["val_loss"], label="Doğrulama Kaybı", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Kayıp (Loss)")
    plt.title("Eğitim ve Doğrulama Kayıpları")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_val_loss.png", dpi=300, bbox_inches='tight')
    plt.close()

def draw_char_confusion_matrices():
    # predictions.json dosyasından gerçek ve tahmin değerlerini oku
    with open("predictions.json", "r") as f:
        preds = json.load(f)
    y_true = []
    y_pred = []
    for p in preds:
        y_true.append(list(str(p["true"]).zfill(6)))
        y_pred.append(list(str(p["predicted"]).zfill(6)))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for i in range(6):
        true_i = y_true[:, i].astype(int)
        pred_i = y_pred[:, i].astype(int)
        cm = confusion_matrix(true_i, pred_i, labels=range(10))
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=[str(i) for i in range(10)], yticklabels=[str(i) for i in range(10)])
        plt.xlabel("Tahmin")
        plt.ylabel("Gerçek")
        plt.title(f"Pozisyon {i+1} Karakter Karışıklık Matrisi")
        plt.savefig(f"char_confusion_matrix_{i+1}.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    save_training_history('training.log', 'training_history.json')
    create_dataset_samples()
    create_model_architecture()
    create_accuracy_plot()
    create_all_custom_diagrams()
    create_additional_diagrams()
    draw_crossentropy_loss_diagram()
    draw_chain_rule_diagram()
    draw_gradient_descent_diagram()
    draw_metrics_box()
    draw_cnn_architecture_block()
    draw_train_val_accuracy()
    draw_train_val_loss()
    draw_char_confusion_matrices()
    print_model_summary()

with open("training_history.json", "r") as f:
    history = json.load(f)

plt.figure(figsize=(10,6))
plt.plot(history["train_loss"], label="Eğitim Kaybı", marker="o")
plt.plot(history["val_loss"], label="Doğrulama Kaybı", marker="s")
plt.title("Eğitim ve Doğrulama Kayıpları")
plt.xlabel("Epoch")
plt.ylabel("Kayıp (Loss)")
plt.legend()
plt.grid(True)
plt.savefig("loss_graph.png")
plt.show()

df = pd.read_csv("captcha_data.csv").sample(5, random_state=42)
model = torch.load("captcha_model.pth")  # veya state_dict yüklenip model.load_state_dict()
summary(model, (1, 50, 200))  # (kanal, yükseklik, genişlik)

for idx, row in df.iterrows():
    img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (200, 50))
    img_tensor = torch.tensor(img_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        outputs = model(img_tensor)
        preds = torch.argmax(outputs, dim=2).squeeze().numpy()
        pred_str = "".join(map(str, preds))
    
    plt.figure()
    plt.imshow(img_resized, cmap="gray")
    plt.title(f"Hedef: {row['solution']} | Tahmin: {pred_str}")
    plt.axis("off")
    plt.savefig(f"sample_{idx}.png")
    plt.close() 