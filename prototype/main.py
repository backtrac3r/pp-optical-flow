import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру!")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Не удалось считать первый кадр!")
        return

    # Переводим в градации серого
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Порог движения (по модулю вектора)
    MAG_THRESHOLD = 5.0

    # Минимальная площадь контуров
    MIN_CONTOUR_AREA = 500

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ЯВНО передаём параметры Farneback
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,   # prev
            gray,        # next
            None,        # flow (None = пусть создаст автоматически)
            0.5,         # pyr_scale
            3,           # levels
            15,          # winsize
            3,           # iterations
            5,           # poly_n
            1.2,         # poly_sigma
            0            # flags
        )

        # Разбиваем flow на горизонтальную и вертикальную составляющие
        fx, fy = flow[..., 0], flow[..., 1]

        # Находим модуль вектора
        mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

        # Создаём бинарную маску (uint8) по порогу
        mask_movement = np.where(mag > MAG_THRESHOLD, 255, 0).astype(np.uint8)

        # Сгладим шум
        mask_movement = cv2.medianBlur(mask_movement, 5)

        # Морфологическое закрытие
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask_movement = cv2.morphologyEx(mask_movement, cv2.MORPH_CLOSE, kernel)

        # Находим контуры
        contours, _ = cv2.findContours(mask_movement, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Соберём bounding boxes и объединим в один
        rects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                rects.append((x, y, w, h))

        if rects:
            total_x1 = min(x for (x, y, w, h) in rects)
            total_y1 = min(y for (x, y, w, h) in rects)
            total_x2 = max(x + w for (x, y, w, h) in rects)
            total_y2 = max(y + h for (x, y, w, h) in rects)

            # Рисуем общий прямоугольник
            cv2.rectangle(frame, (total_x1, total_y1), (total_x2, total_y2), (0, 255, 0), 2)

            # Координаты левого и правого края по центру по вертикали
            midY = (total_y1 + total_y2) // 2
            left_point =  (total_x1, midY)
            right_point = (total_x2, midY)

            # Рисуем точки
            cv2.circle(frame, left_point,  5, (0, 0, 255), -1)
            cv2.circle(frame, right_point, 5, (0, 255, 255), -1)

        cv2.imshow("Webcam (Optical Flow)", frame)
        cv2.imshow("Movement Mask", mask_movement)

        # Обновляем prev_gray
        prev_gray = gray.copy()

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
