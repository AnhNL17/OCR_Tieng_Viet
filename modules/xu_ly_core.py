import os
import platform
import shutil
import cv2
import numpy as np
from PIL import Image
import pytesseract

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN TESSERACT
# ==========================================
if platform.system() == "Windows":
    cac_duong_dan_possible = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.expanduser(r'~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
        os.path.expanduser(r'~\AppData\Local\Tesseract-OCR\tesseract.exe')
    ]
    for path in cac_duong_dan_possible:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
else:
    which_tesseract = shutil.which("tesseract")
    if which_tesseract:
        pytesseract.pytesseract.tesseract_cmd = which_tesseract
    else:
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


# ==========================================
# 2. HÀM TIỀN XỬ LÝ ẢNH
# ==========================================
def xu_ly_anh_truoc_khi_doc(image_input):
    try:
        if isinstance(image_input, Image.Image):
            img_pil = image_input
        elif hasattr(image_input, 'read'):
            image_input.seek(0)
            img_pil = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            img_pil = Image.fromarray(image_input)
        else:
            img_pil = Image.open(image_input)

        if img_pil.mode not in ('RGB', 'L'):
            img_pil = img_pil.convert('RGB')

        img_np = np.array(img_pil)
        img_resized = cv2.resize(img_np, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        if len(img_resized.shape) == 3:
            if img_resized.shape[2] == 3:
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            elif img_resized.shape[2] == 4:
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2GRAY)
            else:
                img_gray = img_resized
        else:
            img_gray = img_resized

        return Image.fromarray(img_gray)
    except Exception:
        return image_input


# ==========================================
# 3. HÀM CHÍNH: NHẬN DẠNG VĂN BẢN (OCR)
# ==========================================
def lay_text_tu_anh(image_file, che_do_doc=3):
    try:
        img_da_xu_ly = xu_ly_anh_truoc_khi_doc(image_file)

        # Lấy trực tiếp thư mục gốc (nơi chứa app.py và file vie.traineddata)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, '..'))

        # Trỏ tessdata-dir trực tiếp về thư mục gốc
        custom_config = f'--tessdata-dir "{root_dir}" --psm {che_do_doc}'

        # Thử nhận dạng bằng model custom v3 (nếu có ở root) hoặc model vie gốc ở root
        try:
            text = pytesseract.image_to_string(img_da_xu_ly, lang='vie_custom_v3', config=custom_config)
        except Exception:
            text = pytesseract.image_to_string(img_da_xu_ly, lang='vie', config=custom_config)

        return text.strip() if text and text.strip() else "Không tìm thấy nội dung văn bản trong ảnh!"

    except Exception as e:
        # Dự phòng cuối cùng: dùng 'vie' mặc định hệ thống
        try:
            text = pytesseract.image_to_string(img_da_xu_ly, lang='vie', config=f'--psm {che_do_doc}')
            return text.strip() if text and text.strip() else "Không tìm thấy nội dung văn bản!"
        except Exception as err:
            return f"Lỗi OCR: {str(err)}"
