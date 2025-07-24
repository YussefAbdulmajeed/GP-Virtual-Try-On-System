import os
import shutil

import subprocess
import sys

def install_package(package_name):
    """Install a package using pip."""
    try:
        # Run the pip install command
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_name}. Error: {e}")

# Install Pillow
install_package("pillow")

from PIL import Image

# الحصول على المسار المطلق
absolute_path = sys.argv[1].strip()

# دالة تغيير حجم الصور
def resize_img(path):
    im = Image.open(path)
    im = im.resize((768, 1024))
    im.save(path)

# تغيير حجم صور الملابس
cloth_dir = os.path.join(absolute_path, 'inputs', 'test', 'cloth')
for path in os.listdir(cloth_dir):
    resize_img(os.path.join(cloth_dir, path))

# الانتقال إلى مجلد المشروع
os.chdir(os.path.join(absolute_path, 'clothes-virtual-try-on'))

# حذف المجلد غير الضروري (.ipynb_checkpoints) إن وجد
checkpoints_path = os.path.join(cloth_dir, '.ipynb_checkpoints')
if os.path.exists(checkpoints_path):
    shutil.rmtree(checkpoints_path)

# تشغيل أكواد المعالجة
os.system(f"python cloth-mask.py {absolute_path}")
os.chdir(absolute_path)
os.system(f"python {absolute_path}clothes-virtual-try-on/remove_bg.py {absolute_path}")

# تشغيل Human Parsing Model
model = os.path.join(absolute_path, "Self-Correction-Human-Parsing", "checkpoints", "final.pth")
input_dir = os.path.join(absolute_path, "inputs", "test", "image")
out_dir = os.path.join(absolute_path, "inputs", "test", "image-parse")
os.system(f"python {absolute_path}Self-Correction-Human-Parsing/simple_extractor.py --dataset lip --model-restore {model} --input-dir {input_dir} --output-dir {out_dir}")

# تشغيل OpenPose
json_dir = os.path.join(absolute_path, "inputs", "test", "openpose-json")
image_dir = os.path.join(absolute_path, "inputs", "test", "openpose-img")
openpose_exe = os.path.join(absolute_path, "openpose", "bin", "OpenPoseDemo.exe")

os.system(f"{openpose_exe} --image_dir {input_dir} --write_json {json_dir} --display 0 --render_pose 0 --hand")
os.system(f"{openpose_exe} --image_dir {input_dir} --display 0 --write_images {image_dir} --hand --render_pose 1 --disable_blending true")

# إنشاء ملف الأزواج بين صور الموديل والملابس
model_images = os.listdir(input_dir)
cloth_images = os.listdir(cloth_dir)
pairs = zip(model_images, cloth_images)

with open(os.path.join(absolute_path, "inputs", "test_pairs.txt"), 'w') as file:
    for model, cloth in pairs:
        file.write(f"{model} {cloth}\n")

# تشغيل سكريبت التنبؤ
os.system(f"python {absolute_path}clothes-virtual-try-on/test.py --name output --dataset_dir {absolute_path}inputs --checkpoint_dir {absolute_path}clothes-virtual-try-on/checkpoints --save_dir {absolute_path}")

# حذف الملفات غير الضرورية
output_checkpoints = os.path.join(absolute_path, "output", ".ipynb_checkpoints")
if os.path.exists(output_checkpoints):
    shutil.rmtree(output_checkpoints)
