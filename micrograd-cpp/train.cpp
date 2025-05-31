#include "train.h"

std::vector<std::shared_ptr<Value>>  convert_values_array(std::vector<double>& x)
{
	std::vector<std::shared_ptr<Value>> new_values;
	new_values.reserve(x.size());
	for (int i = 0; i < x.size(); i++)
	{
		new_values.push_back(std::make_shared<Value>(x[i]));
	}

	return new_values;
}

std::vector<std::vector<std::shared_ptr<Value>>> convert_values_matrix(std::vector<std::vector<double>>& x)
{
	std::vector<std::vector<std::shared_ptr<Value>>> new_values = std::vector<std::vector<std::shared_ptr<Value>>>(x.size());

	for (int i = 0; i < x.size(); i++)
	{
		new_values[i].reserve(x[i].size());
		for (int j = 0; j < x[i].size(); j++)
		{
			new_values[i].push_back(std::make_shared<Value>(x[i][j]));
		}
	}

	return new_values;
}

void save_weights(const char* file_path, std::vector<std::shared_ptr<Value>> wb, int _batch, int _epoch)
{
	std::fstream fout;
	fout.open(file_path, std::ios::out);

	if (!fout.is_open())
	{
		std::cout << "Error: file not opened!";
		return;
	}

	fout << _epoch << "\n";
	fout << _batch << "\n";

	for (int i = 0; i < wb.size();i++)
	{
		fout << wb[i]->data << " ";
	}

	fout.close();

	std::cout << "Succesfully saved weights and biases to " << file_path << "\n";
}

void load_weights(const char* file_path, std::vector<std::shared_ptr<Value>>& wb, int& _batch , int &_epoch)
{
	std::fstream fin;
	fin.open(file_path, std::ios::in);

	if (!fin.is_open())
	{
		std::cout << "Error: file not opened!";
		return;
	}

	fin >> _epoch;
	fin >> _batch;

	for (int i = 0; i < wb.size();i++)
	{
		fin >> wb[i]->data;
	}

	fin.close();

	std::cout << "Succesfully loaded weights and biases from " << file_path << "\n";
}

void softmax(std::vector<std::shared_ptr<Value>>& x)
{
	auto sum_of_exp = std::make_shared<Value>(0.000000001);

	std::vector<std::shared_ptr<Value>> exp_values;

	for (int z = 0; z < x.size(); z++)
	{
		auto exp = x[z]->exp();
		exp_values.push_back(exp);
		sum_of_exp = *sum_of_exp + exp_values[z]; // Accumulate the sum of exp values
	}

	for (int z = 0; z < x.size(); z++)
	{
		auto softmax_value = *exp_values[z] / sum_of_exp;
		x[z] = softmax_value; // Update x[z] to softmax result
	}

	//Calculte jacobian for softmax
	for (int i = 0; i < x.size(); i++)
	{
		x[i]->_backward = [i, x]()
			{
				for (int j = 0; j < x.size(); j++)
				{
					double si = x[i]->data;
					double sj = x[j]->data;
					if (i == j)
						x[i]->grad += si * (1 - si);
					else
						x[i]->grad += -si * sj;
				}
				//printf_s("%f -> %f\n", x[i]->data, x[i]->grad);
			};
	}
}

void sigmoid_output(std::vector<std::shared_ptr<Value>>& x)
{
	for (int z = 0; z < x.size();z++)
	{
		//std::cout << x[z]->data << " : ";
		x[z] = x[z]->sigm();
		//std::cout << x[z]->data << " \n";
	}
}

void tanh_output(std::vector<std::shared_ptr<Value>>& x)
{
	for (int z = 0; z < x.size();z++)
	{
		x[z] = x[z]->tanh();
	}
}

int main()
{
	//srand((unsigned int)time(0)); //for randomness on the initial neuron weights and biases values

	using namespace std::chrono;
	auto start = high_resolution_clock::now();

	std::vector<std::vector<std::string>> data = read_csv("iris_synthetic_data.csv");
	std::vector<std::vector<std::string>> train_data;
	std::vector<std::vector<std::string>> test_data;
	data = shuffle_data(data);
	split_data_80_20(data, train_data, test_data);
	train_data = shuffle_data(train_data);
	std::vector<std::vector<double>> p_data = convert_string_to_double_matrix(train_data);

	data.erase(data.begin(), data.end());//clean data from memory to free up space
	data.shrink_to_fit();

	int batches = p_data.size() / BATCH_SIZE;

	int starting_batch = 0;
	int starting_epoch = 0;

	//system("pause");

	/*neural net*/
	std::vector<int> nn_layers = std::vector<int>{ N_INPUT, 16, 16,  N_OUTPUT };
	MLP nn = MLP(nn_layers);

	MLP::MLP(nn_layers);

	/*weights and biases of the network*/
	std::vector<std::shared_ptr<Value>> wb = nn.get_params();

	std::vector<double> wb_momentum = std::vector<double>(wb.size(),0.0);

	/*training set*/
	std::vector<std::vector<double>> xi = std::vector<std::vector<double>>(BATCH_SIZE, std::vector<double>(N_INPUT, 0.0));

	/*desired output for each item of the training set*/
	std::vector<std::vector<double>> yi = std::vector<std::vector<double>>(BATCH_SIZE, std::vector<double>(N_OUTPUT, 0.0));

	/*converted values*/
	std::vector<std::vector<std::shared_ptr<Value>>> xs = convert_values_matrix(xi);
	std::vector<std::vector<std::shared_ptr<Value>>> ys = convert_values_matrix(yi);

	/*actual output*/
	std::vector<std::shared_ptr<Value>> yt = std::vector<std::shared_ptr<Value>>(ys[0].size());

	std::vector<double> wb_data = std::vector<double>(wb.size());

	double learning_rate = LEARNING_RATE;
	double clip_value = 100.0;

	double momentum_coef = MOMENTUM_COEFFICIENT;

	auto out_size = std::make_shared<Value>(N_OUTPUT);
	auto batch_size = std::make_shared<Value>(BATCH_SIZE);
	double min_loss = 1000;

	/*load weights in order to train the model in multiple sessions*/
	//load_weights("wb.txt", wb, starting_batch, starting_epoch);

	printf_s("starting training with: BATCH_SIZE - %d | EPOCH_SIZE - %d | lr - %f | batches - %d\n", BATCH_SIZE, N_EPOCHS , learning_rate, batches);

	/*training in mini-batches over a few epochs*/
	for (int e = starting_epoch; e < N_EPOCHS; e++)
	{
		train_data = shuffle_data(train_data);
		p_data = convert_string_to_double_matrix(train_data);

		printf_s("epoch %d / %d :\n", e, N_EPOCHS);

		//if (e > 5)
		//{
		//	learning_rate = 0.005;
		//}
		//if (e > 10)
		//{
		//	learning_rate = 0.001;
		//}
		//if (e > 20)
		//{
		//	learning_rate = 0.0005;
		//}
		//if (e > 30)
		//{
		//	learning_rate = 0.0001;
		//}

		for (int b = starting_batch; b < batches  ;b++)
		{

			//printf_s("batch %d / %d :\n", b, batches);

			for (int i = 0; i < BATCH_SIZE; i++)
			{
				int j = i + b * BATCH_SIZE;
				xi[i] = std::vector<double>(p_data[j].size() - 1,  0.0);
				xi[i].assign(p_data[j].begin(), p_data[j].end()-1);
				for (int k = 0; k < xi[i].size(); k++)
				{
					xi[i][k] = xi[i][k];
				}
				//yi[i] = std::vector<double>(N_OUTPUT, -1.0);
				yi[i] = std::vector<double>(N_OUTPUT, 0.0);
				yi[i][(int)p_data[j][4]] = 1.0;
			}


			xs = convert_values_matrix(xi);

			ys = convert_values_matrix(yi);

			/*training process*/

			auto total_loss = std::make_shared<Value>(0.0);

			for (int i = 0; i < xs.size(); i++)
			{
				yt = nn.call(xs[i]);

				for (int j = 0; j < N_OUTPUT; j++)
				{
					/*MSE Loss*/
					auto diff = *yt[j] - ys[i][j];
					auto loss = *diff * diff;
					total_loss = *total_loss + loss;
				}
			}
			
			total_loss = *total_loss / batch_size;

			if(min_loss <= total_loss->data)
			{
				//printf_s("loss: %7.5f | it: %d\n", total_loss->data, b );
			}
			else
			{
				min_loss = total_loss->data;
				printf_s("loss: %7.10f | it: %d -----NEW-LOW-----\n", total_loss->data,  b );
			}

			/*backward pass*/
			total_loss->backward();

			// Add this after backward pass
			bool has_nan = false;
			double max_grad = 0.0;
			double min_grad = 1e9;

			for (int i = 0; i < wb.size(); i++) {
				if (std::isnan(wb[i]->grad)) {
					has_nan = true;
					break;
				}

				max_grad = std::max(max_grad, (wb[i]->grad));
				min_grad = std::min(min_grad, (wb[i]->grad));
			}

			//if (has_nan) {
			//	printf_s("NaN gradient detected! Stopping training.\n");
			//	break;
			//}
			//else if (b % 100 == 0) {
			//	printf_s("Gradient range: [%e, %e]\n", min_grad, max_grad);
			//}

			//#pragma omp parallel for
			for (int i = 0; i < wb.size(); i++)
			{
				if (std::isnan(wb[i]->grad)) {
					//std::cout << "NaN gradient detected!" << i << std::endl;
					printf_s("NaN gradient detected!");
					return -1;
				}
				wb[i]->grad = std::max(std::min(wb[i]->grad, clip_value), -clip_value);
				
				/*gradient descent wthout momentum*/
				//wb[i]->data += -learning_rate * wb[i]->grad;

				/*gradient descent with another proposed momentum (https://distill.pub/2017/momentum/)*/
				wb_momentum[i] = momentum_coef * wb_momentum[i] + wb[i]->grad;
				wb[i]->data = wb[i]->data - (learning_rate * wb_momentum[i]);
			}

			for (int i = 0; i < wb.size(); i++)
			{
				wb[i]->grad = 0.0;
			}

			if (b % 100 == 0)
			{
				/*applying softmax for easier interpretability*/
				//softmax(yt);

				/*saving current weights every batch*/
				save_weights("wb.txt", wb, b + 1, e);

				printf_s("loss: %7.5f | it: %d \n", total_loss->data, b);

				for (int i = 0; i < N_OUTPUT;i++)
				{
					printf_s("guess: %.2f | label: %.2f | %d\n", yt[i]->data, ys[BATCH_SIZE - 1][i]->data, i);
				}
			}
		}
		starting_batch = 0;
		/*and every epoch*/
		save_weights("wb.txt", wb, 0, e+1);
	}

	double accuracy = 0;
	double t_poz = 0;
	double t_neg = 0;

	///*testing process*/
	test_data = shuffle_data(test_data);
	p_data.clear();
	p_data.shrink_to_fit();
	p_data = convert_string_to_double_matrix(test_data);
	batches = p_data.size() / BATCH_SIZE;

	for (int b = 0; b < batches; b++)
	{
		for (int i = 0; i < BATCH_SIZE; i++)
		{
			int j = i + b * BATCH_SIZE;
			xi[i] = std::vector<double>(p_data[j].size() - 1, 0.0);
			xi[i].assign(p_data[j].begin(), p_data[j].end() - 1);
			for (int k = 0; k < xi[i].size(); k++)
			{
				xi[i][k] = xi[i][k];
			}
			//yi[i] = std::vector<double>(N_OUTPUT, -1.0);
			yi[i] = std::vector<double>(N_OUTPUT, 0.0);
			yi[i][(int)p_data[j][4]] = 1.0;
		}


		xs = convert_values_matrix(xi);
		ys = convert_values_matrix(yi);

		/*testing process*/
		for (int i = 0; i < xs.size(); i++)
		{
			yt = nn.call(xs[i]);
			//testing

			int ypred = 0;
			int ycorr = 0;

			for (int j = 1; j < ys[i].size(); j++)
			{
				if (yt[ypred]->data < yt[j]->data)
				{
					ypred = j;
				}

				if (ys[i][j]->data == 1)
				{
					ycorr = j;
				}
			}

			if (ypred == ycorr)
			{
				t_poz++;
			}
			else
			{
				t_neg++;
			}
		}
	}
	
	accuracy = t_poz / (t_poz + t_neg);
	printf_s("\naccuracy: %f\n", accuracy);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);

	printf_s("execution time: %Id microseconds\n", duration.count());

	return 0;
}
