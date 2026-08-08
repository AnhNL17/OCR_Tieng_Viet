import os
import platform
import shutil
import cv2
import numpy as np
from PIL import Image
import pytesseract

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN TESSERACT LINH HOẠT
# ==========================================
if platform.system() == "Windows":
    # Danh sách các đường dẫn cài đặt Tesseract phổ biến trên Windows
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
            
    # Nếu không tìm thấy ở các đường dẫn cố định, tìm trong PATH hệ thống
    if not tesseract_found:
        which_tesseract = shutil.which("tesseract")
        if which_tesseract:
            pytesseract.pytesseract.tesseract_cmd = which_tesseract
else:
    # Trên Linux / Streamlit Cloud: Tự động tìm lệnh 'tesseract' trong hệ thống
    which_tesseract = shutil.which("tesseract")
    if which_tesseract:
        pytesseract.pytesseract.tesseract_cmd = which_tesseract
    else:
        # Mặc định đường dẫn chuẩn của tesseract trên Ubuntu/Linux
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


# ==========================================
# 2. HÀM TIỀN XỬ LÝ ẢNH
# ==========================================
def xu_ly_anh_truoc_khi_doc(image_file):
    """
    Hàm tiền xử lý: Đọc ảnh -> Phóng to Bicubic -> Chuyển ảnh xám (Grayscale)
    """
    try:
        # Tự động đọc ảnh nếu truyền vào là chuỗi đường dẫn hoặc file uploader
        if isinstance(image_file, (str, bytes)):
            img = Image.open(image_file)
        else:
            img = image_file
            
        img = np.array(img)

        # Phóng to ảnh gấp 2 lần để tăng độ nét cho OCR
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Chuyển ảnh về dạng xám nếu là ảnh màu
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        return Image.fromarray(img)
    except Exception:
        # Nếu gặp lỗi format, trả lại ảnh gốc
        return image_file


# ==========================================
# 3. HÀM CHÍNH: NHẬN DẠNG VĂN BẢN (OCR)
# ==========================================
def lay_text_tu_anh(image_file, che_do_doc=3):
    """
    Hàm thực hiện OCR trích xuất văn bản từ ảnh/PDF
    """
    try:
        # 1. Tiền xử lý ảnh
        img_da_xu_ly = xu_ly_anh_truoc_khi_doc(image_file)

        # 2. Xác định đường dẫn thư mục chứa model traineddata
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_model = os.path.abspath(os.path.join(current_dir, '..', 'Train', 'Model'))

        # Chuẩn bị chuỗi config
        custom_config = f'--tessdata-dir "{path_to_model}" --psm {che_do_doc}'
        
        text = ""

        # 3. Thử nhận dạng bằng Model Custom (vie_custom_v3)
        try:
            text = pytesseract.image_to_string(
                img_da_xu_ly, 
                lang='vie_custom_v3', 
                config=custom_config
            )
        except Exception:
            # 4. FALLBACK: Nếu không tìm thấy vie_custom_v3 hoặc lỗi tessdata-dir, 
            # tự động chuyển sang dùng model 'vie' mặc định của hệ thống
            config_fallback = f'--psm {che_do_doc}'
            text = pytesseract.image_to_string(
                img_da_xu_ly, 
                lang='vie', 
                config=config_fallback
            )

        # 5. Kiểm tra kết quả trả về
        if text and text.strip():
            return text.strip()
        else:
            return "Không tìm thấy nội dung văn bản trong ảnh!"

    except Exception as e:
        return f"Lỗi OCR: {str(e)}"
