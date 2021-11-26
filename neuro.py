import numpy as np

class Neuro:
    '''
      Нейронная сеть с одним скрытым слоем из 2 нейронов, одним выходом и двумя входами
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

    
    def __init__(self):
        '''
            Конструктор без параметров
            
            Задает случайные значения
            для весов и смещений
        '''
        # Начальные случайные веса
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Начальные случайные смещения
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()


    def feed_forward(self, x, get_type = 0):
        '''
            Функция переднего прохода
            
            Вычисляет значения спрятанных и выходного нейронов
            
            Параметр get_type отвечает за набор возвращаемых значений
            0 - (по умолчанию) возврат только значения выходного нейрона
            1 - (для внутреннего использования) - возврат всех
            вычисляемых значений в виде словаря
        '''
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = self.sigm(sum_h1)
        
        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = self.sigm(sum_h2)
        
        sum_out = self.w5 * h1 + self.w6 * h2 + self.b3
        output = self.sigm(sum_out)
        
        if (get_type == 1):
            return {'sum_h1': sum_h1, 'sum_h2': sum_h2, 'sum_out': sum_out,
                    'h1': h1, 'h2': h2, 'output': output}
        elif (get_type == 0):
            return output


    def train(self, data, all_y_trues):
        '''
            Функция тренировки
            
            Отрабатывает определенное количество итераций
            передний проход и изменение весов в соответствии с его результатами
        '''
        iterations = 1000

        for iteration in range(iterations):
          for x, y_true in zip(data, all_y_trues):  
            inter_res = self.feed_forward(x, 1)
            
            sum_h1 = inter_res['sum_h1']
            h1 = inter_res['h1']
            
            sum_h2 = inter_res['sum_h2']
            h2 = inter_res['h2']

            sum_out = inter_res['sum_out']
            output = inter_res['output']
            y_pred = output

            self.change_params(x, y_true, y_pred, h1, h2, sum_h1, sum_h2, sum_out)


    def change_params(self, x, y_true: list, y_pred: list, h1: float, h2: float,
                      sum_h1: float, sum_h2: float, sum_out: float):
        '''
            Функция изменения весов и смещений

            Принимает на вход результаты последнего
            переднего прохода

            Рассматриваем ф-цию потерь как ф-цию от весов и смещений
            L(w1, w2, w3, w4, w5, w6, b1, b2, b3)
        '''
        learn_rate = 0.1
        
        # Вычисление частных производных методом обратного распространения ошибок
        # d_L_d_ypred значит dL по dypred
        d_L_d_ypred = -2 * (y_true - y_pred)

        # output
        d_ypred_d_w5 = h1 * self.sigm_deriv(sum_out)
        d_ypred_d_w6 = h2 * self.sigm_deriv(sum_out)
        d_ypred_d_b3 = self.sigm_deriv(sum_out)

        d_ypred_d_h1 = self.w5 * self.sigm_deriv(sum_out)
        d_ypred_d_h2 = self.w6 * self.sigm_deriv(sum_out)

        # h1
        d_h1_d_w1 = x[0] * self.sigm_deriv(sum_h1)
        d_h1_d_w2 = x[1] * self.sigm_deriv(sum_h1)
        d_h1_d_b1 = self.sigm_deriv(sum_h1)

        # h2
        d_h2_d_w3 = x[0] * self.sigm_deriv(sum_h2)
        d_h2_d_w4 = x[1] * self.sigm_deriv(sum_h2)
        d_h2_d_b2 = self.sigm_deriv(sum_h2)

        # Меняем веса и смещения
        # h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # output
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
