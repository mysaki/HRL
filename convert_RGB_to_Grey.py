import cv2
def convert_to_grayscale_and_save(image_path, output_path):
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return
        # 转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 保存灰度图像
        cv2.imwrite(output_path, gray_image)
        print(f"已将 {image_path} 转换为灰度图像并保存到 {output_path}")
    except Exception as e:
        print(f"处理图像时出现错误: {e}")


# 示例使用
file_path = 'D:/XNW/程序/Unet/input/turtlebot4_sim/images'
image_path_1 = "observe_27.jpg"
output_path_1 = "gray_image1.jpg"
image_path_2 = "observe_30.jpg"
output_path_2 = "gray_image2.jpg"

convert_to_grayscale_and_save(image_path_1, output_path_1)
convert_to_grayscale_and_save(image_path_2, output_path_2)
