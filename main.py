import cv2

def main():
    # Инициализируем камеру (0 - первый индекс веб-камеры)
    cap = cv2.VideoCapture(0)

    # Создаём объект субтракции фона
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    # Пороговая площадь контура (минимальный размер области для определения движения)
    MIN_CONTOUR_AREA = 500

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Применяем фоновую субтракцию
        fg_mask = backSub.apply(frame)

        # Сглаживаем шумы в маске
        fg_mask = cv2.medianBlur(fg_mask, 5)

        # Находим контуры
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Обходим найденные контуры
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Проверяем, достаточно ли большой контур
            if area > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)

                # Координаты левого и правого края (по горизонтали)
                left_point  = (x, y + h // 2)
                right_point = (x + w, y + h // 2)

                # Рисуем рамку вокруг двигающегося объекта
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Рисуем точки на левом и правом краю
                cv2.circle(frame, left_point, 5, (0, 0, 255), -1)    # Красная точка
                cv2.circle(frame, right_point, 5, (0, 255, 255), -1) # Желтая точка

        # Показываем оригинальное видео и маску движения
        cv2.imshow('Webcam', frame)
        cv2.imshow('FG Mask', fg_mask)

        # Выход по нажатию ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
