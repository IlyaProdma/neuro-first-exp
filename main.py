from neuro import Neuro
import numpy as np
import warnings
from image_work import image_to_array

def input_data(filename: str):
    # Читаем датасет из файла
    f = open(filename, 'r')
    num_cols = len(f.readline().split())
    f.close()
    data = np.loadtxt(filename, usecols=(np.arange(0,num_cols-1)))
    results = np.loadtxt(filename, usecols=(num_cols-1))
    # Смещения параметров на средние значения
    num_cols = np.shape(data)[1]
    biases = [data[:,i].mean() for i in range(num_cols)]
    for i in range(num_cols):
        data[:,i] -= biases[i]
    return {'data': data, 'results': results, 'biases': biases}

def input_image(filename: str):
    data = np.loadtxt(filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # Запись данных
    filename = input("Введите название файла с тренировочными данными: ")
    response = input_data(filename)
    data = response['data']
    results = response['results']
    biases = response['biases']

    # Тренировка сети
    network = Neuro(256, 10)
    network.train(data, results, 1000)

    class1 = input("Введите название типа объектов, которым соответствует 1: ")
    class2 = input("Введите название типа объектов, которым соответствует 0: ")

    # Проверка на новых значениях
    while True:
        try:
            filename = input("Введите название файла с данными для определения: ")
            if (filename.split('.')[1] == 'jpg'):
                new_data = image_to_array(filename)
            else:
                new_data = np.loadtxt(filename)
            num_cols = len(new_data)
            new_data = [new_data[i] - biases[i] for i in range(num_cols)]
            output = network.feed_forward(new_data)
            print(f"Результат: {output:.3f}")
            #print(f"Результат1: {output_values[1]:.3f}")
            class_res = class1 if output > 0.5 else class2
            print(f"Объект больше соответствует классу '{class_res}'")
        except:
            print("Произошла ошибка, проверьте название файла")

    
