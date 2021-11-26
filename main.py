from neuro import Neuro
import numpy as np

def input_data(filename: str):
    # Читаем датасет из файла
    data = np.loadtxt(filename, usecols=(0,1))
    results = np.loadtxt(filename, usecols=(2))

    # Смещения параметров на средние значения
    bias_first = data[:,0].mean()
    bias_second = data[:,1].mean()
    data[:,0] -= bias_first
    data[:,1] -= bias_second

    return {'data': data, 'results': results, 'b1': bias_first, 'b2': bias_second}

if __name__ == '__main__':
    # Запись данных
    filename = input("Введите название файла с тренировочными данными: ")
    response = input_data(filename)
    data = response['data']
    results = response['results']
    bias_first = response['b1']
    bias_second = response['b2']

    # Тренировка сети
    network = Neuro()
    network.train(data, results)

    class1 = input("Введите название типа объектов, которым соответствует 1: ")
    class2 = input("Введите название типа объектов, которым соответствует 0: ")

    # Проверка на новых значениях
    param_first = float(input("Введите первый параметр объекта: ")) - bias_first
    param_second = float(input("Введите второй параметр объекта: ")) - bias_second
    output = network.feed_forward(np.array([param_first, param_second]))
    print(f"Результат: {output:.3f}")
    class_res = class1 if output > 0.5 else class2
    print(f"Объект больше соответствует классу '{class_res}'")

    
