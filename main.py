import os
import cv2
from tqdm import tqdm

# Niqobni oddiy kvadrat shaklida pastki yuzga chizamiz
def apply_mask(image):
    h, w = image.shape
    mask_color = 60  # quyuq kulrang
    # pastki yarmi maskalanadi
    image[h//2:] = mask_color
    return image

# Tasvirlarni yuklab, niqobli versiyasini saqlash
def process_images(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    emotions = os.listdir(source_dir)

    for emotion in emotions:
        emotion_src_path = os.path.join(source_dir, emotion)
        emotion_dst_path = os.path.join(target_dir, emotion)
        os.makedirs(emotion_dst_path, exist_ok=True)

        for img_name in tqdm(os.listdir(emotion_src_path), desc=emotion):
            img_path = os.path.join(emotion_src_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            masked_img = apply_mask(img)
            cv2.imwrite(os.path.join(emotion_dst_path, img_name), masked_img)

# Asosiy ishni bajarish
if __name__ == "__main__":
    process_images("train", "masked_train")
    process_images("test", "masked_test")
    print("✅ Barcha tasvirlarga niqob qo‘yildi.")
