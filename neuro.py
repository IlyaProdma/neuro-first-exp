import numpy as np

class Neuron:
    def __init__(self, num_input):
        self.weights = [np.random.normal() for i in range(num_input)]
        self.bias = np.random.normal()

class Neuro:
    '''
      Нейронная сеть с одним скрытым слоем из 2 нейронов, одним выходом и неограниченными входами...
    '''
    
    def sigm(self, x: float) -> float:
        ''' Сигмоида '''
        return 1 / (1 + np.exp(-x))


    def sigm_deriv(self, x: float) -> float:
        ''' Производная сигмоиды '''
        return self.sigm(x) * (1 - self.sigm(x))

    
    def mse(self, y_true: np.array, y_pred: np.array) -> float:
        ''' Средняя квадратическая ошибка '''
        return ((y_true - y_pred)**2).mean()
    
    def __init__(self, num_input, num_hidden):
        '''
            Конструктор без параметров
            
            Задает случайные значения
            для весов и смещений
        '''
        # Начальные случайные веса для скрытого и выходного слоев
        # и начальные случайные смещения для скрытого слоя
        self.hidden_neurons = [Neuron(num_input) for i in range(num_hidden)]
        self.weights_out = [np.random.normal() for i in range(num_hidden)]
        
        # Начальное случайное смещение выходного слоя
        self.b_out = np.random.normal()


    def feed_forward(self, x, get_type = 0):
        '''
            Функция переднего прохода
            
            Вычисляет значения спрятанных и выходного нейронов
            
            Параметр get_type отвечает за набор возвращаемых значений
            0 - (по умолчанию) возврат только значения выходного нейрона
            1 - (для внутреннего использования) - возврат всех
            вычисляемых значений в виде словаря
        '''
        hidden_sums = [0 for i in range(len(self.hidden_neurons))]
        hidden_values = [0 for i in range(len(self.hidden_neurons))]
        for i in range(len(self.hidden_neurons)):
            for j in range(len(self.hidden_neurons[i].weights)):
                hidden_sums[i] += self.hidden_neurons[i].weights[j] * x[j]
            hidden_sums[i] += self.hidden_neurons[i].bias
            hidden_values[i] = self.sigm(hidden_sums[i])
        
        sum_out = 0
        for i in range(len(self.weights_out)):
            sum_out += self.weights_out[i] * hidden_values[i]
        sum_out += self.b_out
        output = self.sigm(sum_out)
        
        if (get_type == 1):
            return {'hidden_sums': hidden_sums, 'sum_out': sum_out,
                    'hidden_values': hidden_values, 'output': output}
        elif (get_type == 0):
            return output


    def train(self, data, all_y_trues, iterations = 1000):
        '''
            Функция тренировки
            
            Отрабатывает определенное количество итераций
            передний проход и изменение весов в соответствии с его результатами
        '''
        for iteration in range(iterations):
            for x, y_true in zip(data, all_y_trues):
                inter_res = self.feed_forward(x, 1)
                hidden_sums = inter_res['hidden_sums']
                hidden_values = inter_res['hidden_values']

                sum_out = inter_res['sum_out']
                y_pred = inter_res['output']
                self.change_params(x, y_true, y_pred, hidden_values, hidden_sums, sum_out)
            if (iteration % 100 == 0 and iteration > 0):
                print("iteration: ", iteration)


    def change_params(self, x, y_true: list, y_pred: list, hidden_values: list,
                      hidden_sums: list, sum_out: float):
        '''
            Функция изменения весов и смещений

            Принимает на вход результаты последнего
            переднего прохода

            Рассматриваем ф-цию потерь как ф-цию от весов и смещений
            L(wi, bj)
        '''
        learn_rate = 0.1
        
        # Вычисление частных производных методом обратного распространения ошибок
        # d_L_d_ypred значит dL по dypred
        d_L_d_ypred = -2 * (y_true - y_pred)

        # output
        d_ypred_d_wouts = [hidden_values[i]*self.sigm_deriv(sum_out) for i in range(len(hidden_values))]
        d_ypred_d_b_out = self.sigm_deriv(sum_out)


        # Меняем веса и смещения
        # скрытый слой
        for i in range(len(self.hidden_neurons)):
            d_ypred_d_hi = self.weights_out[i] * self.sigm_deriv(sum_out)
            for j in range(len(self.hidden_neurons[i].weights)):
                d_hi_d_wj = x[j] * self.sigm_deriv(hidden_sums[i])
                self.hidden_neurons[i].weights[j] -= learn_rate * d_L_d_ypred * d_ypred_d_hi * d_hi_d_wj
            d_hi_d_bi = self.sigm_deriv(hidden_sums[i])
            self.hidden_neurons[i].bias -= learn_rate * d_L_d_ypred * d_ypred_d_hi * d_hi_d_bi
        
        # выходной нейрон
        for i in range(len(self.weights_out)):
            self.weights_out[i] -= learn_rate * d_L_d_ypred * d_ypred_d_wouts[i]
        self.b_out -= learn_rate * d_L_d_ypred * d_ypred_d_b_out
        
