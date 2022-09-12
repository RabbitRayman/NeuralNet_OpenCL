//активационная функция
double BetaHyperbolicTanActivationFunction(double beta,
										double gain,
										double input)
{
	double res = (exp(2.0 * beta * input) - 1.0) / (exp(2.0 * beta * input) + 1.0);
	res = res * gain;
	return res;
}

//представление нейронной сети с использованием одного массива,
//в котором хранятся значения в нейронах для всех слоев, кроме последнего
//на вход - массив слоёв (количество нейронов в них), массив синаптических весов и смещений, массив входных значений для нейронки и массив,
//для сохранения выходных значений нейронки
void NeuralNetType1(__constant uint* layers,
					__constant double* weights,
					__constant double* z,
					__global double* out)
{
    //массив нейронов
	double allNeurons[nNeurons] = {0};
	//запись входных значений в первый слой
	for (int i = 0; i<num_of_inputs; i++)
	{
		allNeurons[i] = z[i];
	}
	//индекс первого веса в текущей итерации
	int firstW = 0;
	//индекс первого нейрона в текущей итерации
	int firstNeuron = 0;
	//индекс первого смещения в текущей итерации
	int firstBias = 0;
	//цикл для всех слоев, кроме первого (он уже задан)
	for (int i = 1; i < nLayers; i++)
	{
	    //цикл для всех нейронов в слое
		for (int l = 0; l<layers[i];l++)
		{
		    //сумма значений поступающих в нейрон
			double sum = 0;
			//суммирование значений, поступающих в нейрон
			for (int k = 0; k<layers[i-1];k++)
			{
				sum = sum + allNeurons[firstNeuron+k]*weights[firstW+k+l*layers[i-1]];
			}
			//если последний слой, то значение активациационной функции записывается в выходной массив
			if (i == nLayers-1)
			{
				out[l] = BetaHyperbolicTanActivationFunction(2.0, 1.1, (sum+weights[SizeOfWeigths - (nBias - (firstBias+l))]));
			}
			//иначе записывается во внутренний массив
			else
			{
				allNeurons[firstNeuron+layers[i-1]+l] = BetaHyperbolicTanActivationFunction(2.0, 1.0, (sum+weights[SizeOfWeigths - (nBias - (firstBias+l))]));
			}
		}
		//следующие значения для индексов
		firstW = firstW + layers[i]*layers[i-1];
		firstNeuron = firstNeuron + layers[i-1];
		firstBias = firstBias + layers[i];
	}
}

//представление нейронной сети с использованием двух массивов
//(каждый из которых равен размеру самого большого слоя)
//в которых хранятся значения в нейронах для двух слоев, прошлый и текущий
//на вход - массив слоёв (количество нейронов в них), массив синаптических весов и смещений, массив входных значений для нейронки и массив,
//для сохранения выходных значений нейронки
void NeuralNetType2(__constant uint* layers,
					__constant double* weights,
					__constant double* z,
					__global double* out)
{
    //создание массива прошлого слоя
	double prevNeurons[maxLayer] = {0};
	//запись в массив прошлого слоя входных значений
	for (int i = 0; i<num_of_inputs; i++)
	{
		prevNeurons[i] = z[i];
	}
	//индекс первого веса в текущей итерации
	int firstW = 0;
	//индекс первого смещения в текущей итерации
	int firstBias = 0;
	//цикл для всех слоев, кроме первого (он уже задан)
	for (int i = 1; i < nLayers; i++)
	{
	    //создание массива текущего слоя
		double currNeurons[maxLayer] = {0};
		//цикл для всех нейронов в слое
		for (int l = 0; l<layers[i];l++)
		{
		    //сумма значений поступающих в нейрон
			double sum = 0;
			//суммирование значений, поступающих в нейрон
			for (int k = 0; k<layers[i-1];k++)
			{
				sum = sum + prevNeurons[k]*weights[firstW+k+l*layers[i-1]];
			}
			//если последний слой, то значение активациационной функции записывается в выходной массив
			if (i == nLayers-1)
			{
				out[l] = BetaHyperbolicTanActivationFunction(2.0, 1.1, (sum+weights[SizeOfWeigths - (nBias - (firstBias+l))]));
			}
			//иначе записывается в массив текущего слоя
			else
			{
				currNeurons[l] = BetaHyperbolicTanActivationFunction(2.0, 1.0, (sum+weights[SizeOfWeigths - (nBias - (firstBias+l))]));
			}
		}
		//если слой не последний, значения из массива текущего слоя переписываются в массив прошлого слоя
		//для следующей итерации. Также обновляются значения индексов
		if (i != nLayers-1)
		{
			firstW = firstW + layers[i]*layers[i-1];
			for (int l = 0; l<layers[i];l++)
			{
				prevNeurons[l] = currNeurons[l];
			}
			firstBias = firstBias + layers[i];
		}
	}
}


//kernel-ядро, выполняющее запуск реализации нейронной сети в рамках потока
//кол-во потоков = размер популяции
//на входе - массив для выходных значений результата работы нейронки, массив синаптических весов для нейронки, массив входных значений для нейронки,
//массив, содержащий количество нейронов в каждом слое, массив ожидаемых результатов работы нейронки
__kernel void CalculateError(__global double* q,
						__constant double* w,
						__constant double* z,
						__constant uint* layers,
						__constant double* expect)
{
	//проверка на номер потока
	int global_id = get_global_id(0);
	if (global_id >= nPopulation) 
	{
		return;
	}
	//цикл для всех наборов входных значений
	for (int i = 0; i < num_z; i++)
	{
		//указание на запуск одной из двух реализаций нейронки (вероятно, можно сделать это без выбора на каждой итерации, чтобы ускорить работу)
		if (NeuralType == 1)
			NeuralNetType1(&layers[0],&w[global_id*SizeOfWeigths],&z[i*num_of_inputs],&q[(global_id*num_z*num_of_outputs)+(i*num_of_outputs)]);
		else if (NeuralType == 2)
			NeuralNetType2(&layers[0],&w[global_id*SizeOfWeigths],&z[i*num_of_inputs],&q[(global_id*num_z*num_of_outputs)+(i*num_of_outputs)]);
		//цикл для заданного количества выходных значений работы нейронки (см. выходной(==последний) слой нейронки)
		for (int j = 0; j < num_of_outputs; j++)
		{
			//подсчёт и сохранение квадратичной ошибки работы нейронки - из ожидаемого значения вычитается полученное и результат возводится в квадрат
			//для каждого выходного значения выполняется отдельно
			q[(global_id*num_z*num_of_outputs)+(i*num_of_outputs)+j] = pow((expect[(i*num_of_outputs)+j] - q[(global_id*num_z*num_of_outputs)+(i*num_of_outputs)+j]),2);
		}
	}
}

//kernel-ядро, выполняющее поиск максимальной ошибки работы нейронки для каждого её выходного значения (см. выходной(==последний) слой нейронки) в рамках потока
//кол-во потоков = размер популяции
//на входе - массив выходных значений результата работы нейронки, массив для записи максимальных ошибок
__kernel void maximum(__constant double* q,
                    __global double* v)
{
	//проверка на номер потока
    int global_id = get_global_id(0);
    if (global_id >= nPopulation)
    {
        return;
    }
	//цикл для заданного количества выходных значений работы нейронки (см. выходной(==последний) слой нейронки)
    for (int k = 0; k<num_of_outputs;k++)
    {
		//подсчёт индекса первого из выходных значений работы нейронки для данного потока и данной итерации
        int index = global_id * num_z * num_of_outputs + k;
		//инициализация переменной для хранения максимальной ошибки
        double max = q[index];
		//цикл по количеству наборов входных данных нейронки
        for (int i = 1; i<num_z;i++)
        {
			//вычисление индекса для выходного значения работы нейронки
            index = index + num_of_outputs;
			//сравнение с хранимым максимумом
            if (max < q[index])
            {
                max=q[index];
            }
        }
		//запись в массив максимальных ошибок
        v[(global_id * num_of_outputs) + k] = max;
    }
}

//kernel-функция
bool fitness_logic(double a, double b)
{
	return (a>=b);
}


//kernel-ядро, выполняющее подсчёт значения фитнесс-функции (в данном случае - через множество Парето) для каждой особи
//на входе - массив для записи значений фитнесс-функции, массив входных значений - максимальные ошибки, найденные в maximum (см. выше), 
//количество особей (размер популяции)
__kernel void paretoFitness(__global double* fit,
							__constant double* criteriaValues,
							int number)
{
	//проверка на номер потока
	int global_id = get_global_id(0);
    if (global_id >= number)
    {
        return;
    }
	//если размер популяции = 1, то сразу же присваиваем единственной особи значение фитнесс-функции, равное 1
	if(nPopulation == 1)
	{
	   fit[0] = 1;
	}
	else
	{
		//переменная для подсчёта точек, входящих в конус доминирования
		double count = 0;
		//массив, содержащий в себе значения для заданной особи (каждый поток - своя отдельная особь)
		double repmat[num_of_outputs] = {0};
		//переменная для подсчёта точек, лежащих внутри конуса доминирования
		double d1 = 0;
		//переменная для подсчёта точек, лежащих прямо на исследуемой особи
		double d2 = 0;
		//переменная для подсчёта количества особей (может быть, можно избавиться и использовать переменную number)
		double num = 0;
		//запись в массив значений заданной особи
		for (int j=0;j<num_of_outputs;j++)
		{
			repmat[j]=criteriaValues[(global_id*num_of_outputs)+j];
		}
		//запуск цикла, в рамках которого происходит поиск и подсчёт точек, входящих в конус доминирования
		for (int k = 0; k<number;k++)
		{
			//переменная для подсчёта критериев особи, которые находятся за пределами конуса доминирования
			double c1 = 0;
			//переменная для подсчёта критериев особи, которые находятся на конусе доминирования
			double c2 = 0;
			//цикл для 
			for (int j=0;j<num_of_outputs;j++)
			{
				//переменная, которая демонстрирует нахождение критерия особи цикла k относительно критерия особи потока
				double temp = repmat[j]-criteriaValues[(k*num_of_outputs)+j];
				//если temp находится внутри критерия
				c1 = c1 + fitness_logic(temp,0);
				//если temp равен критерию особи потока
				if (temp==0)
					c2 = c2 + 1;
			}
			//увеличение счётчка особей
			num = num + 1;
			//если все критерии особи k находятся в конусе доминирования особи потока
			if (c1 == num_of_outputs)
				d1 = d1 + 1;
			//если все критерии особи k находятся на особи потока
			if (c2 == num_of_outputs)
				d2 = d2 + 1;
		}
		//убираем из особей, которые находятся в конусе доминирования, особи, которые находятся на особи потока
		count = d1 - d2;
		//считаем значение фитнесс-функции; степень используется для усиления (?)
		fit[global_id] = pow( ( 1.0 + count/(num-1.0)),(-Q) );
	}
}

//kernel-ядро, реализующее селекцию родителей, используя метод турнира
//на входе - массив значений фитнесс-функций, массив для выходных значений - индексы родителей, массив индексов, из которых составляются пары для турнира
__kernel void selectionTournament(__constant double* population_fitness,
								__global uint* parentsIndexes,
								__constant uint* global_indexes)
{
	//проверка на номер потока
	int global_id = get_global_id(0);
    if (global_id >= nPopulation)
    {
        return;
    }
	//массив, в котором хранится пара значений фитнесс-функций от родителей, которые участвуют в турнире
	double tourParticipant[2] = {0};
	//запись значений фитнесс-функций в массив
	for (int i = 0; i<2;i++)
	{
		tourParticipant[i]=population_fitness[global_indexes[(global_id*2)+i]];
	}
	//сравнение участников и сохранение индекса наибольшего из них в массив родителей
    if ( tourParticipant[0]>=tourParticipant[1] )
		parentsIndexes[global_id] = global_indexes[(global_id*2)];
	else
		parentsIndexes[global_id] = global_indexes[(global_id*2)+1];
}


//kernel-функция, реализующая кроссовер
//на входе - массив значений фитнесс-функций, массив для выходных значений - индексы родителей, 
//массив индексов, из которых составляются пары для турнира
void realSBXCrossover(__constant double* parent1,
					__constant double* parent2,
					double* child1,
					double* child2,
					__constant ushort* k,
					__constant double* rand)
{
	for (int i = 0; i<k[0]; i++)
	{
		child1[i]=parent1[i];
		child2[i]=parent2[i];
	}
	
	for (int i = k[0]; i<dimensions; i++)
	{
		child1[i]=parent2[i];
		child2[i]=parent1[i];
	}
    
    double u = rand[0];
	
	double beta = 0;
	
    if(u<=0.5)
        beta = pow((2.0*u),(1.0/(crossover_param+1.0)));
    else
        beta = pow((1.0/(2.0*(1.0-u))),(1.0/(crossover_param+1.0)));
  
    child1[k[0]] = 0.5*((1.0+beta)*parent1[k[0]] + (1.0-beta)*parent2[k[0]]);
    child2[k[0]] = 0.5*((1.0-beta)*parent1[k[0]] + (1.0+beta)*parent2[k[0]]); 
}


//kernel-функция, реализующая мутацию
//на входе - 
void realRandomMutation(__constant ushort* r,
						double* child,
						__constant short* limits,
						__constant double* rand)
{
	double limit1 = limits[0];
	double limit2 = limits[1];
	child[r[0]] = limit1 + rand[0]*(limit2-limit1);
}


//kernel-ядро, реализующее создание новой популяции
//на входе - массив текущей популяции, массив для сохранения новой популяции, массив индексов родителей, 
//два массива со случайными значениями, массив ограничений значений в популяции
__kernel void newPopulation(__constant double* population,
							__global double* pNextPopulation,
							__constant uint* parents,
							__constant double* rand, 
							__constant ushort* rand_k, 
							__constant short* limits)
{
	//проверка на номер потока
	int global_id = get_global_id(0);
    if (global_id >= (nPopulation/2))
    {
        return;
    }
	//два массива для потомков заданной пары родителей, размер каждого из которых (массивов потомков) равен количеству синаптических весов и смещений
	double child1[dimensions] = {0};
	double child2[dimensions] = {0};
	//запись начальный индексов синаптических весов и смещений родителей в переменные
	int parent1 = parents[global_id*2]*dimensions;
	int parent2 = parents[(global_id*2)+1]*dimensions;
	//путём сравнения рандомного числа и заданного вручную шанса кроссовера, определяем проводить ли кроссовер или просто приравнять потомков к родителям
	if (rand[global_id*5]<=change_crossover)
		realSBXCrossover(&population[parent1],&population[parent2],&child1[0],&child2[0],&rand_k[global_id*3],&rand[(global_id*5)+3]);
	else
	{
		for (int i = 0; i < dimensions; i++)
		{
			child1[i]=population[parent1+i];
			child2[i]=population[parent2+i];
		}
	}
	//путём сравнения рандомного числа и заданного вручную шанса мутации, определяем проводить ли мутацию потомка
	if (rand[(global_id*5)+1] <= change_mutation)
		realRandomMutation(&rand_k[(global_id*3)+1],&child1[0],&limits[0],&rand[(global_id*5)+4]);

	if (rand[(global_id*5)+2] <= change_mutation)
		realRandomMutation(&rand_k[(global_id*3)+2],&child2[0],&limits[0],&rand[(global_id*5)+4]);
	
	//сохраняем получившихся потомков в массив новой популяции
	for (int i=0;i<dimensions;i++)
	{
		pNextPopulation[(global_id*2*dimensions)+i] = child1[i];
		pNextPopulation[(((global_id*2)+1)*dimensions)+i] = child2[i];
	}
}
