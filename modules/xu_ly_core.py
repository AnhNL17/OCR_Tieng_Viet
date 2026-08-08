import pytesseract
from PIL import Image
import cv2 
import numpy as np
import os
import platform

# Tự động kiểm tra hệ điều hành
if platform.system() == "Windows":
    # Nếu chạy dưới máy cá nhân Windows thì mới dùng đường dẫn ổ C
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    # Trên Linux / Streamlit Cloud: Không gán đường dẫn cứng, 
    # hệ thống tự nhận diện Tesseract từ packages.txt
    pass
def xu_ly_anh_truoc_khi_doc(image_file):
    """
    Hàm tiền xử lý: Đọc ảnh -> Phóng to -> Chuyển đen trắng
    """
    img = Image.open(image_file)
    img = np.array(img)
    
    # Phóng to ảnh gấp 2 lần
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Chuyển về đen trắng
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return Image.fromarray(img)

def lay_text_tu_anh(image_file, che_do_doc=3):
    try:
        # Tiền xử lý ảnh
        img_da_xu_ly = xu_ly_anh_truoc_khi_doc(image_file)

        # Trỏ đến thư mục chứa model (Train/Model)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_model = os.path.abspath(os.path.join(current_dir, '..', 'Train', 'Model'))

        # Thiết lập biến môi trường để Tesseract tự nhận diện thư mục data (Tránh lỗi khoảng trắng đường dẫn)
        os.environ['TESSDATA_PREFIX'] = path_to_model

        # Cấu hình PSM và ngôn ngữ
        config_v3 = f'--psm {che_do_doc} -l vie_custom_v3'
        
        # Chạy OCR
        text = pytesseract.image_to_string(img_da_xu_ly, config=config_v3)
        
        return text if text.strip() else "Không đọc được chữ nào cả!"
        
    except Exception as e:
        return f"Lỗi: {str(e)}"
