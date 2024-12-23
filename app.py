import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import io
from skimage.morphology import erosion, dilation, opening, closing, skeletonize, white_tophat, black_tophat, disk, rectangle, diamond, star, octagon, ellipse
from skimage.filters import gabor, gaussian
from skimage.transform import resize, rotate, AffineTransform, warp, ProjectiveTransform, pyramid_gaussian, rescale
from skimage import exposure
import pywt
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



def morph_operation(image, kernel, operation):
    if operation == "Erosion":
        return erosion(image, kernel)
    elif operation == "Dilation":
        return dilation(image, kernel)
    elif operation == "Opening":
        return opening(image, kernel)
    elif operation == "Closing":
        return closing(image, kernel)
    elif operation == "Gradient":
        return dilation(image, kernel) - erosion(image, kernel)
    elif operation == "Top-hat trắng":
        return white_tophat(image.astype(np.uint8), kernel)
    elif operation == "Top-hat đen":
        return black_tophat(image.astype(np.uint8), kernel)


def main():
    st.set_page_config(layout="wide")  # Set layout to wide
    st.title("Top 1 xử lý ảnh")

    # Sidebar for file upload and parameter adjustments
    with st.sidebar:
        st.header("Tùy chọn xử lý ảnh")
        uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image = np.array(image)

                # Validate image format
                if image.dtype != np.uint8:
                    st.error("Ảnh phải có kiểu dữ liệu uint8 (8-bit).")
                    return

                if image.ndim not in [2, 3]:
                    st.error("Ảnh phải là ảnh xám (2 chiều) hoặc ảnh màu (3 chiều).")
                    return

                # Convert RGBA to RGB if necessary
                if image.ndim == 3 and image.shape[2] == 4:
                    image = image[:, :, :3]

                option = st.selectbox("Chọn thuật toán", (
                    "Điều chỉnh độ sáng và độ tương phản", 
                    "Biến đổi không gian màu", 
                    "Gamma Correction",
                    "Biến đổi Fourier", "Biến đổi Wavelet", "Phát hiện biên Sobel",
                    "Phát hiện biên Canny", "Biến đổi Hough", "Ngưỡng hóa ảnh", "Phân cụm K-means",
                    "Bộ lọc Gabor", "Các phép toán hình thái học", "Bộ xương ảnh", "Cân bằng Histogram",
                    "Làm sắc nét ảnh (Unsharp Masking)", "Làm sắc nét với Laplacian", "Thay đổi kích thước ảnh", "Xoay ảnh",
                    "Biến đổi Affine", "Biến đổi phối cảnh", "Biến đổi Pyramid", "Lọc trung bình",
                    "Lọc trung vị", "Lọc Gaussian", "Lọc song phương", "Phân tích kết cấu LBP",
                    "Top-hat trắng", "Top-hat đen", "Dịch chuyển ảnh", "Cắt ảnh", "Lật ảnh", "Color Quantization",
                ))

                adjusted_image = image.copy()  # Initialize adjusted_image for download and preview

                if option == "Điều chỉnh độ sáng và độ tương phản":
                    brightness = st.slider("Độ sáng", -100, 100, 0)
                    contrast = st.slider("Độ tương phản", -100, 100, 0)
                    adjusted_image = cv2.convertScaleAbs(image, alpha=1 + contrast / 100, beta=brightness)

                elif option == "Biến đổi không gian màu":
                    color_space = st.selectbox("Chọn không gian màu", ( "HSV", "LAB", "Grayscale"))
                    if color_space == "HSV":
                        adjusted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    elif color_space == "LAB":
                        adjusted_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                    elif color_space == "Grayscale":
                        adjusted_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    else:
                        adjusted_image = image

                elif option == "Gamma Correction":
                    gamma = st.slider("Gamma", 0.1, 3.0, 1.0)
                    adjusted_image = np.array(255 * (image / 255) ** gamma, dtype='uint8')

                elif option == "Biến đổi Fourier":
                    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    f_transform = np.fft.fft2(gray_image)
                    f_shift = np.fft.fftshift(f_transform)
                    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
                    adjusted_image = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                elif option == "Biến đổi Wavelet":
                    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    coeffs2 = pywt.dwt2(gray_image, 'haar')
                    LL, (LH, HL, HH) = coeffs2
                    adjusted_image = np.hstack([
                        exposure.rescale_intensity(LL, out_range=(0, 255)).astype(np.uint8),
                        exposure.rescale_intensity(LH, out_range=(0, 255)).astype(np.uint8),
                        exposure.rescale_intensity(HL, out_range=(0, 255)).astype(np.uint8),
                        exposure.rescale_intensity(HH, out_range=(0, 255)).astype(np.uint8),
                    ])

                elif option == "Phát hiện biên Sobel":
                    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
                    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
                    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
                    adjusted_image = exposure.rescale_intensity(sobel_combined, out_range=(0, 255)).astype(np.uint8)

                elif option == "Phát hiện biên Canny":
                    threshold1 = st.slider("Ngưỡng thấp", 0, 255, 50)
                    threshold2 = st.slider("Ngưỡng cao", 0, 255, 150)
                    adjusted_image = cv2.Canny(image, threshold1, threshold2)

                elif option == "Biến đổi Hough":
                    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray_image, 50, 150)
                    rho = st.slider("Rho", 1, 10, 1)
                    theta = st.slider("Theta (độ)", 1, 180, 90)
                    threshold = st.slider("Ngưỡng", 50, 200, 100)
                    lines = cv2.HoughLines(edges, rho, np.radians(theta), threshold)
                    adjusted_image = image.copy()
                    if lines is not None:
                        for line in lines:
                            rho, theta = line[0]
                            a = np.cos(theta)
                            b = np.sin(theta)
                            x0 = a * rho
                            y0 = b * rho
                            x1 = int(x0 + 1000 * (-b))
                            y1 = int(y0 + 1000 * a)
                            x2 = int(x0 - 1000 * (-b))
                            y2 = int(y0 - 1000 * a)
                            cv2.line(adjusted_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                elif option == "Ngưỡng hóa ảnh":
                    threshold = st.slider("Ngưỡng", 0, 255, 127)
                    _, adjusted_image = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), threshold, 255, cv2.THRESH_BINARY)

                elif option == "Phân cụm K-means":
                    k = st.slider("Số cụm", 2, 10, 3)
                    data = image.reshape((-1, 3)).astype(np.float32)
                    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
                    clustered = kmeans.cluster_centers_[kmeans.labels_]
                    adjusted_image = clustered.reshape(image.shape).astype(np.uint8)

                elif option == "Bộ lọc Gabor":
                    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    theta = st.slider("Theta", 0.0, np.pi, 0.0)
                    frequency = st.slider("Tần số", 0.1, 1.0, 0.6)
                    filtered, _ = gabor(gray_image, frequency=frequency, theta=theta)
                    adjusted_image = exposure.rescale_intensity(filtered, out_range=(0, 255)).astype(np.uint8)

                elif option == "Các phép toán hình thái học":
                    operation = st.selectbox("Chọn phép toán", ("Erosion", "Dilation", "Opening", "Closing", "Gradient", "Top-hat trắng", "Top-hat đen"))
                    kernel_shape = st.selectbox("Hình dạng kernel", ("Vuông", "Chữ nhật", "Diamond", "Star", "Octagon"))
                    kernel_size = st.slider("Kích thước kernel", 1, 21, 3, 2)

                    if kernel_shape == "Vuông":
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    elif kernel_shape == "Chữ nhật":
                        kernel = rectangle(kernel_size, kernel_size * 2)
                    elif kernel_shape == "Diamond":
                        kernel = diamond(kernel_size // 2)
                    elif kernel_shape == "Star":
                        kernel = star(kernel_size // 2)
                    elif kernel_shape == "Octagon":
                        kernel = octagon(kernel_size // 2, kernel_size // 3)

                    if image.ndim == 3:  # For RGB images
                        adjusted_image = np.zeros_like(image)
                        for channel in range(3):
                            adjusted_image[:, :, channel] = morph_operation(image[:, :, channel], kernel, operation)
                    else:  # For grayscale images
                        adjusted_image = morph_operation(image, kernel, operation)
                        
                elif option == "Bộ xương ảnh":
                    # Chuyển đổi ảnh sang ảnh xám nếu cần
                    if image.ndim == 3:
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_image = image

                    # Tiền xử lý (tùy chọn)
                    preprocessing = st.checkbox("Áp dụng tiền xử lý (Lọc Gaussian)")
                    if preprocessing:
                        sigma = st.slider("Sigma Gaussian", 0.1, 5.0, 1.0)
                        gray_image = cv2.GaussianBlur(gray_image, (5, 5), sigma)

                    threshold = st.slider("Ngưỡng", 0, 255, 127)
                    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)


                    # Áp dụng thuật toán bộ xương
                    adjusted_image = skeletonize(binary_image / 255)

                    # Xử lý hậu kỳ (tùy chọn)
                    postprocessing = st.checkbox("Áp dụng xử lý hậu kỳ (Làm mảnh)")
                    if postprocessing:
                        kernel = np.ones((3,3), np.uint8)
                        adjusted_image = cv2.dilate(adjusted_image.astype(np.uint8), kernel, iterations=1) #膨胀


                elif option == "Cân bằng Histogram":
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    adjusted_image = cv2.equalizeHist(gray_image)
                    
                    
                elif option == "Làm sắc nét ảnh (Unsharp Masking)":
                    blur = cv2.GaussianBlur(image, (21, 21), 10)
                    adjusted_image = cv2.addWeighted(image, 1.5, blur, -0.5, 0)
                   
                elif option == "Xoay và dịch chuyển ảnh":
                    angle = st.slider("Góc xoay", -180, 180, 0)
                    scale = st.slider("Tỉ lệ", 0.1, 2.0, 1.0)
                    center = (image.shape[1] // 2, image.shape[0] // 2)
                    M = cv2.getRotationMatrix2D(center, angle, scale)
                    adjusted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                  
                  
                elif option == "Biến đổi Affine":
                    # Nhập tọa độ các điểm nguồn
                    st.write("Nhập tọa độ các điểm nguồn (x, y):")
                    pt1_x = st.number_input("Điểm 1 - x", value=50)
                    pt1_y = st.number_input("Điểm 1 - y", value=50)
                    pt2_x = st.number_input("Điểm 2 - x", value=200)
                    pt2_y = st.number_input("Điểm 2 - y", value=50)
                    pt3_x = st.number_input("Điểm 3 - x", value=50)
                    pt3_y = st.number_input("Điểm 3 - y", value=200)


                    # Nhập tọa độ các điểm đích
                    st.write("Nhập tọa độ các điểm đích (x, y):")
                    pt1_x_dst = st.number_input("Điểm 1 - x (đích)", value=10)
                    pt1_y_dst = st.number_input("Điểm 1 - y (đích)", value=100)
                    pt2_x_dst = st.number_input("Điểm 2 - x (đích)", value=200)
                    pt2_y_dst = st.number_input("Điểm 2 - y (đích)", value=50)
                    pt3_x_dst = st.number_input("Điểm 3 - x (đích)", value=100)
                    pt3_y_dst = st.number_input("Điểm 3 - y (đích)", value=250)

                    # Tạo ma trận biến đổi
                    src_points = np.float32([[pt1_x, pt1_y], [pt2_x, pt2_y], [pt3_x, pt3_y]])
                    dst_points = np.float32([[pt1_x_dst, pt1_y_dst], [pt2_x_dst, pt2_y_dst], [pt3_x_dst, pt3_y_dst]])
                    M = cv2.getAffineTransform(src_points, dst_points)

                    # Áp dụng biến đổi
                    adjusted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


                elif option == "Biến đổi phối cảnh":
                    # Nhập tọa độ các điểm nguồn
                    st.write("Nhập tọa độ các điểm nguồn (x, y):")
                    pt1_x = st.number_input("Điểm 1 - x", value=56)
                    pt1_y = st.number_input("Điểm 1 - y", value=65)
                    pt2_x = st.number_input("Điểm 2 - x", value=450)
                    pt2_y = st.number_input("Điểm 2 - y", value=54)
                    pt3_x = st.number_input("Điểm 3 - x", value=28)
                    pt3_y = st.number_input("Điểm 3 - y", value=360)
                    pt4_x = st.number_input("Điểm 4 - x", value=470)
                    pt4_y = st.number_input("Điểm 4 - y", value=370)

                    # Nhập tọa độ các điểm đích
                    st.write("Nhập tọa độ các điểm đích (x, y):")
                    pt1_x_dst = st.number_input("Điểm 1 - x (đích)", value=0)
                    pt1_y_dst = st.number_input("Điểm 1 - y (đích)", value=0)
                    pt2_x_dst = st.number_input("Điểm 2 - x (đích)", value=400)
                    pt2_y_dst = st.number_input("Điểm 2 - y (đích)", value=0)
                    pt3_x_dst = st.number_input("Điểm 3 - x (đích)", value=0)
                    pt3_y_dst = st.number_input("Điểm 3 - y (đích)", value=300)
                    pt4_x_dst = st.number_input("Điểm 4 - x (đích)", value=400)
                    pt4_y_dst = st.number_input("Điểm 4 - y (đích)", value=300)

                    # Tạo ma trận biến đổi
                    src_points = np.float32([[pt1_x, pt1_y], [pt2_x, pt2_y], [pt3_x, pt3_y], [pt4_x, pt4_y]])
                    dst_points = np.float32([[pt1_x_dst, pt1_y_dst], [pt2_x_dst, pt2_y_dst], [pt3_x_dst, pt3_y_dst], [pt4_x_dst, pt4_y_dst]])

                    M = cv2.getPerspectiveTransform(src_points, dst_points)

                    # Áp dụng biến đổi
                    adjusted_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

   
                elif option == "Biến đổi Pyramid":
                    adjusted_image = cv2.pyrDown(image)
              
                elif option == "Lọc trung bình":
                    ksize = st.slider("Kích thước kernel", 1, 15, 5, 2)
                    adjusted_image = cv2.blur(image, (ksize, ksize))
                
                elif option == "Lọc trung vị":
                    ksize = st.slider("Kích thước kernel", 1, 15, 5, 2)
                    adjusted_image = cv2.medianBlur(image, ksize)
                 
                 
                elif option == "Lọc Gaussian":
                    sigma = st.slider("Sigma", 0.1, 5.0, 1.0)
                    adjusted_image = cv2.GaussianBlur(image, (15, 15), sigma)
               
                elif option == "Lọc song phương":
                    adjusted_image = cv2.bilateralFilter(image, 15, 75, 75)
              
                elif option == "Phân tích kết cấu LBP":
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')

                    # Chuẩn hóa LBP về khoảng 0-255
                    adjusted_image = exposure.rescale_intensity(lbp, out_range=(0, 255)).astype(np.uint8)


                elif option == "Top-hat trắng": # Ví dụ về Top-hat trắng
                    kernel_size = st.slider("Kích thước kernel", 1, 21, 3, 2)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    adjusted_image = white_tophat(gray_image, kernel)
                   

                elif option == "Top-hat đen": # Ví dụ về Top-hat đen
                    kernel_size = st.slider("Kích thước kernel", 1, 21, 3, 2)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    adjusted_image = black_tophat(gray_image, kernel)
                    
                    
                elif option == "Thay đổi kích thước ảnh":
                        width = st.number_input("Chiều rộng", min_value=1, value=image.shape[1])
                        height = st.number_input("Chiều cao", min_value=1, value=image.shape[0])
                        adjusted_image = cv2.resize(image, (width, height))
                       
                elif option == "Xoay ảnh":
                    angle = st.slider("Góc xoay", -180, 180, 0)
                    rows, cols = image.shape[:2]
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                    adjusted_image = cv2.warpAffine(image, M, (cols, rows))
                 
                elif option == "Dịch chuyển ảnh":
                    tx = st.number_input("Dịch chuyển theo x", value=0.0, step=0.1) # Cho phép nhập số thập phân
                    ty = st.number_input("Dịch chuyển theo y", value=0.0, step=0.1)

                    rows, cols = image.shape[:2]
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    adjusted_image = cv2.warpAffine(image, M, (cols, rows))

                    # Hiển thị ảnh gốc và ảnh đã dịch chuyển cạnh nhau
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Ảnh gốc", use_column_width=True)
                    with col2:
                        st.image(adjusted_image, caption="Ảnh đã dịch chuyển", use_column_width=True)
            
            
                elif option == "Cắt ảnh":
                    x1 = st.number_input("x1", value=0)
                    y1 = st.number_input("y1", value=0)
                    x2 = st.number_input("x2", value=image.shape[1])
                    y2 = st.number_input("y2", value=image.shape[0])

                    adjusted_image = image[y1:y2, x1:x2]
                 
                elif option == "Lật ảnh":
                    flip_axis = st.radio("Chọn trục lật", ("Theo trục x", "Theo trục y", "Cả hai trục"))
                    if flip_axis == "Theo trục x":
                        adjusted_image = cv2.flip(image, 0)
                    elif flip_axis == "Theo trục y":
                        adjusted_image = cv2.flip(image, 1)
                    else:
                        adjusted_image = cv2.flip(image, -1)
                  

                elif option == "Color Quantization":
                    n_colors = st.slider("Số lượng màu", 2, 256, 8)
                    pixels = image.reshape((-1, 3))
                    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(pixels)
                    quantized_image = kmeans.cluster_centers_[kmeans.labels_]
                    adjusted_image = quantized_image.reshape(image.shape).astype(np.uint8)
                

                elif option == "Làm sắc nét với Laplacian":
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
                    adjusted_image = exposure.rescale_intensity(np.abs(laplacian), out_range=(0, 255)).astype(np.uint8)  # Chuẩn hóa để hiển thị

                # Allow user to download the edited image
                st.markdown("---")
                st.subheader("Tải ảnh đã chỉnh sửa")
                result = Image.fromarray(adjusted_image)
                buf = io.BytesIO()
                result.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Đao ảnh nhé!",
                    data=byte_im,
                    file_name="adjusted_image.png",
                    mime="image/png"
                )

            except Exception as e:
                st.error(f"Lỗi xử lý ảnh: {e}")
                return
        else:
            st.warning("Vui lòng tải lên ảnh để bắt đầu xử lý.")

    # Display original and adjusted images side by side outside the sidebar
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Ảnh gốc", use_column_width=True)
        with col2:
            st.image(adjusted_image, caption="Ảnh đã chỉnh sửa", use_column_width=True)


if __name__ == "__main__":
    main()