import os
import platform
import shutil
import cv2
import numpy as np
from PIL import Image
import pytesseract

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN TESSERACT LINH HOẠT THEO HỆ ĐIỀU HÀNH
# ==========================================
if platform.system() == "Windows":
    # Các đường dẫn cài đặt Tesseract phổ biến trên Windows
    cac_duong_dan_possible = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.expanduser(r'~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'),
        os.path.expanduser(r'~\AppData\Local\Tesseract-OCR\tesseract.exe')
    ]
    
    tesseract_found = False
    for path in cac_duong_dan_possible:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            break
            
    if not tesseract_found:
        which_tesseract = shutil.which("tesseract")
        if which_tesseract:
            pytesseract.pytesseract.tesseract_cmd = which_tesseract
else:
    # Trên Linux / Streamlit Cloud: Tự động tìm binary tesseract trong hệ thống
    which_tesseract = shutil.which("tesseract")
    if which_tesseract:
        pytesseract.pytesseract.tesseract_cmd = which_tesseract
    else:
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


# ==========================================
# 2. HÀM TIỀN XỬ LÝ ẢNH (CHỐNG LỖI UNSUPPORTED IMAGE)
# ==========================================
def xu_ly_anh_truoc_khi_doc(image_input):
    """
    Tiền xử lý ảnh: Chuẩn hóa mọi định dạng về PIL Image,
    phóng to Bicubic x2 và chuyển sang ảnh xám (Grayscale).
    """
    try:
        # A. Ép kiểu dữ liệu đầu vào về PIL Image
        if isinstance(image_input, Image.Image):
            img_pil = image_input
        elif hasattr(image_input, 'read'):  # UploadedFile từ Streamlit hoặc BytesIO
            image_input.seek(0)
            img_pil = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            img_pil = Image.fromarray(image_input)
        else:
            img_pil = Image.open(image_input)

        # Chuyển đổi mode ảnh bất thường về RGB hoặc L
        if img_pil.mode not in ('RGB', 'L'):
            img_pil = img_pil.convert('RGB')

        # B. Chuyển sang NumPy array để tiền xử lý bằng OpenCV
        img_np = np.array(img_pil)

        # C. Phóng to ảnh gấp 2 lần (Bicubic) tăng độ rõ nét cho chữ
        img_resized = cv2.resize(img_np, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # D. Chuyển sang ảnh xám (Grayscale)
        if len(img_resized.shape) == 3:
            if img_resized.shape[2] == 3:
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            elif img_resized.shape[2] == 4:
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2GRAY)
            else:
                img_gray = img_resized
        else:
            img_gray = img_resized

        # E. Trả về PIL Image chuẩn cho pytesseract
        return Image.fromarray(img_gray)

    except Exception:
        # Nếu gặp lỗi xử lý OpenCV, trả lại ảnh PIL nguyên bản
        try:
            if hasattr(image_input, 'seek'):
                image_input.seek(0)
            return Image.open(image_input)
        except Exception:
            return image_input


# ==========================================
# 3. HÀM CHÍNH: NHẬN DẠNG VĂN BẢN (OCR)
# ==========================================
def lay_text_tu_anh(image_file, che_do_doc=3):
    """
    Thực hiện trích xuất văn bản từ ảnh/PDF bằng Tesseract OCR.
    """
    try:
        # 1. Tiền xử lý ảnh an toàn
        img_da_xu_ly = xu_ly_anh_truoc_khi_doc(image_file)

        # 2. Xác định đường dẫn thư mục và file model custom
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_model_dir = os.path.abspath(os.path.join(current_dir, '..', 'Train', 'Model'))
        custom_model_file = os.path.join(path_to_model_dir, 'vie_custom_v3.traineddata')

        text = ""

        # 3. Kiểm tra sự tồn tại thực tế của file model custom v3
        if os.path.exists(custom_model_file):
            # Nếu có file custom: Dùng --tessdata-dir trỏ đến Train/Model và load vie_custom_v3
            custom_config = f'--tessdata-dir "{path_to_model_dir}" --psm {che_do_doc}'
            text = pytesseract.image_to_string(
                img_da_xu_ly, 
                lang='vie_custom_v3', 
                config=custom_config
            )
        else:
            # Nếu KHÔNG có file custom: Chạy bằng 'vie' hệ thống Linux (cài từ packages.txt)
            # KHÔNG truyền --tessdata-dir để tránh lỗi hỏng đường dẫn
            config_fallback = f'--psm {che_do_doc}'
            text = pytesseract.image_to_string(
                img_da_xu_ly, 
                lang='vie', 
                config=config_fallback
            )

        # 4. Trả về kết quả
        if text and text.strip():
            return text.strip()
        else:
            return "Không tìm thấy nội dung văn bản trong ảnh!"

    except Exception as e:
        # Cơ chế bảo vệ cuối cùng (Fallback an toàn)
        try:
            config_fallback = f'--psm {che_do_doc}'
            text = pytesseract.image_to_string(
                img_da_xu_ly, 
                lang='vie', 
                config=config_fallback
            )
            return text.strip() if text.strip() else "Không tìm thấy nội dung văn bản!"
        except Exception as err:
            return f"Lỗi OCR: {str(err)}"
