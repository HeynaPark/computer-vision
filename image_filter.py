import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    # 이미지를 넘파이 배열로 로드
    image = Image.open(image_path)
    image = np.array(image)
    return image

def show_image(image):
    # 이미지를 화면에 표시
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def apply_filter(image, kernel):
    # 이미지에 커널을 적용하여 필터링
    filtered_image = np.zeros_like(image)
    for i in range(3):  # RGB 채널에 대해 각각 필터링
        filtered_image[:, :, i] = np.abs(np.convolve(image[:, :, i].ravel(), kernel.ravel(), mode='same').reshape(image.shape[:2]))
    return np.clip(filtered_image, 0, 255).astype(np.uint8)  # 결과를 0~255 범위로 클리핑 후 정수형으로 변환


def main():
    # 이미지 로드
    image_path = 'example_img.jpg'
    original_image = load_image(image_path)
    
    # 가로 선 필터링 커널 정의
    horizontal_kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    
    # 이미지에 가로 선 필터링 적용
    filtered_image = apply_filter(original_image, horizontal_kernel)
    
    # 원본 이미지와 필터링된 이미지 한 번에 표시
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(filtered_image)
    axes[1].set_title('Filtered Image')
    axes[1].axis('off')
    plt.show()

if __name__ == "__main__":
    main()


