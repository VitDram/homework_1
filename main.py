import cv2
import numpy as np
import pymysql
import requests
import base64
import time

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_encoded_image

user_input = input("Введите текстовые данные: ")

connection = pymysql.connect(
    host='server141.hosting.reg.ru',
    user='u2483709_default',
    password='4VKfLDW6FkRc338n',
    database='u2483709_default'
)

cursor = connection.cursor()
sql = "INSERT INTO `base64 images` (`image`) VALUES (%s)"
cursor.execute(sql, (user_input,))
connection.commit()
connection.close()
print("Данные успешно добавлены в базу данных.C:/Users/user/Desktop/Program file_1/Program file/assets/Processed_Chernyaevka.jpg")

def draw_object_bounding_box(image_to_process, index, box):
    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 1  # уменьшенная ширина рамки
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    start = (x, y - 5)
    font_size = 0.5  # уменьшенный размер шрифта
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 1  # уменьшенная ширина текста
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font, font_size, color, width, cv2.LINE_AA)

    return final_image

def draw_class_count(image_to_process, class_name, count, start_position):
    font_size = 0.4  # уменьшенный размер шрифта
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 1  # уменьшенная ширина текста
    text = f"{class_name} count: {count}"

    # вывод текста с обводкой (чтобы было видно при разном освещении картинки)
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image_to_process, text, start_position, font, font_size, black_outline_color, width * 2, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start_position, font, font_size, white_color, width, cv2.LINE_AA)

    return final_image

def apply_yolo_object_detection(image_to_process):
    """
    Распознавание и определение координат объектов на изображении
    :param image_to_process: исходное изображение
    :return: изображение с размеченными объектами и подписями к ним
    """
    height, width, depth = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    class_indexes, class_scores, boxes = ([] for i in range(3))
    car_count = 0
    person_count = 0

    # запуск поиска объектов на изображении
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)

                box = [center_x - obj_width // 2, center_y - obj_height // 2, obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # проведение выборки
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    
    if isinstance(chosen_boxes, np.ndarray) and chosen_boxes.ndim > 1:  # если chosen_boxes многомерный массив
        for box_index_array in chosen_boxes:
            box_index = box_index_array[0]
            box = boxes[box_index]
            class_index = class_indexes[box_index]

             # Подсчет количества машин и людей
            if classes[class_index] == "car":
               car_count += 1
            elif classes[class_index] == "person":
               person_count += 1

            if classes[class_index] in classes_to_look_for:
              image_to_process = draw_object_bounding_box(image_to_process, class_index, box)


    elif isinstance(chosen_boxes, np.ndarray):  # если chosen_boxes одномерный массив
        for box_index in chosen_boxes:
            box = boxes[box_index]
            class_index = class_indexes[box_index]
            # Подсчет количества машин и людей
            if classes[class_index] == "car":
                car_count += 1
            elif classes[class_index] == "person":
               person_count += 1

            if classes[class_index] in classes_to_look_for:
               image_to_process = draw_object_bounding_box(image_to_process, class_index, box)

    # Отображение количества машин и людей
    final_image = draw_class_count(image_to_process, "Car", car_count, (45, 50))
    final_image = draw_class_count(final_image, "Person", person_count, (45, 80))

    if person_count > 3:
        final_image = cv2.putText(final_image, "You should wait, there are a lot of people at the border.", (45, 110),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)  # уменьшенный размер и ширина текста
    if car_count > 3:
        final_image = cv2.putText(final_image, "You should wait, there's a lot of traffic at the border.", (45, 130),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)  # уменьшенный размер и ширина текста

    return final_image, car_count, person_count

def start_image_object_detection():
    """
    Анализ изображения и отправка в базу данных
    """
    while True:
        try:
            # применение методов распознавания объектов на изображении от YOLO
            image_path = "assets/moscow.jpg"
            image = cv2.imread(image_path)
            final_image, car_count, person_count = apply_yolo_object_detection(image)

            # вывод результатов на консоль
            print(f"Car count: {car_count}")
            print(f"Person count: {person_count}")

            if person_count > 3:
                print("You should wait, there are a lot of people at the border.")
            if car_count > 3:
                print("You should wait, there's a lot of traffic at the border.")

            # сохранение обработанного изображения
            processed_image_path = "assets/Processed_moscow.jpg"
            cv2.imwrite(processed_image_path, final_image)

            # конвертация обработанного изображения в Base64
            base64_image = encode_image_to_base64(processed_image_path)

            # отправка изображения в базу данных
            connection = pymysql.connect(
                host='server141.hosting.reg.ru',
                user='u2483709_default',
                password='4VKfLDW6FkRc338n',
                database='u2483709_default'
            )
            cursor = connection.cursor()
            sql = "INSERT INTO `base64 images` (`image`) VALUES (%s)"
            cursor.execute(sql, (base64_image,))
            connection.commit()
            connection.close()
            print("Изображение успешно добавлено в базу данных.")

            # вывод обработанного изображения на экран
            cv2.imshow("Image", final_image)
            if cv2.waitKey(0):
                cv2.destroyAllWindows()

            # задержка на 2 минуты перед повторением
            time.sleep(120)

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    # загрузка весов YOLO из файлов и настройка сети
    net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()

    # Проверка типа и преобразование, если необходимо
    if isinstance(out_layers_indexes, np.ndarray):
      out_layers_indexes = out_layers_indexes.tolist()  # преобразование в список
    if all(isinstance(item, int) for item in out_layers_indexes): # проверка на int значения
      out_layers = [layer_names[i - 1] for i in out_layers_indexes]  # приводим к списку
    else:
      out_layers = [layer_names[index[0] - 1] for index in out_layers_indexes] # если index это list

    # загрузка из файла классов объектов, которые умеет обнаруживать YOLO
    with open("coco.names.txt") as file:
        classes = file.read().split("\n")

    # определение классов, которые будут приоритетными для поиска на изображении
    # названия находятся в файле "coco.names.txt"
    classes_to_look_for = ["truck", "person", "car"]

    start_image_object_detection()
