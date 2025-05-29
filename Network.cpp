#include "Network.h"

std::shared_ptr<Value> rand_val(int n_in)
{
	std::random_device dev;
	std::mt19937_64 rng(dev());
	// Xavier/Glorot initialization
	double limit = sqrt(2.0 / n_in);
	std::uniform_real_distribution<> dist(-limit, limit);
	return std::make_shared<Value>(dist(rng));
}

Neuron::Neuron()
{
	std::cout << "initialized empty neuron\n";
}

Neuron::Neuron(int n_in, bool _apply_activation = true)
{
	apply_activation = _apply_activation;
	w = std::vector<std::shared_ptr<Value>>(n_in);
	for (int i = 0; i < w.size(); i++)
	{
		w[i] = rand_val(n_in);
	}
	b = rand_val(n_in);
}

std::shared_ptr<Value> Neuron::call(std::vector<std::shared_ptr<Value>>& x)
{
	std::shared_ptr<Value> out = std::make_shared<Value>(0.0);

	for (int i = 0; i < w.size(); i++)
	{
		out = *(*w[i] * x[i]) + out;
	}

	out = *out + b;

	//out->print();

	if (apply_activation)
	{
		out = out->tanh();
		//out = out->sigm();
		//out = out->relu();
		//out = out->leaky_relu();
		//out = out->elu();//doesnt work
	}

	//out->print();

	return out; //buna vlad sunt bibi
}

Layer::Layer()
{
	std::cout << "initialized empty layer\n";
}

Layer::Layer(int n_in, int n_out, bool _apply_activation)
{
	neurons = std::vector<std::shared_ptr<Neuron>>(n_out);
	for (int i = 0; i < n_out; i++)
	{
		neurons[i] = std::make_shared<Neuron>(n_in, _apply_activation);
	}
}

std::vector<std::shared_ptr<Value>> Layer::call(std::vector<std::shared_ptr<Value>>& x)
{
	std::vector<std::shared_ptr<Value>> outs = std::vector<std::shared_ptr<Value>>(neurons.size());
	for (int i = 0; i < neurons.size(); i++)
	{
		outs[i] = neurons[i]->call(x);
	}
	return outs;
}

MLP::MLP()
{
	std::cout << "initialized empty MLP\n";
}

MLP::MLP(std::vector<int>& _layers)
{
	layers = std::vector<std::shared_ptr<Layer>>(_layers.size() - 1);
	for (int i = 0; i < _layers.size() - 1; i++)
	{
		//excludint the application of the activation at the end to have a broader range for softmax values
		//layers[i] = std::make_shared<Layer>(_layers[i], _layers[i + 1], (i < _layers.size() - 2));
		layers[i] = std::make_shared<Layer>(_layers[i], _layers[i + 1], true);
	}
}

std::vector<std::shared_ptr<Value>> MLP::call(std::vector<std::shared_ptr<Value>>& x)
{
	std::vector<std::shared_ptr<Value>> outs = x;
	for (int i = 0; i < layers.size(); i++)
	{
		outs = layers[i]->call(outs);
	}
	return outs;
}

std::vector<std::shared_ptr<Value>> MLP::get_params()
{
	std::vector<std::shared_ptr<Value>> wb;
	for (int i = 0; i < layers.size(); i++)
	{
		for (int j = 0; j < layers[i]->neurons.size(); j++)
		{
			for (int k = 0; k < layers[i]->neurons[j]->w.size();k++)
			{
				wb.push_back(layers[i]->neurons[j]->w[k]);
			}
			wb.push_back(layers[i]->neurons[j]->b);
		}
	}

	return wb;
}


// Ensure proper cleanup
void MLP::reset()
{
	for (auto& layer : layers)
	{
		for (auto& neuron : layer->neurons)
		{
			// Explicitly reset shared_ptr of neuron weights/biases if needed.
			for (auto& weight : neuron->w)
			{
				weight.reset();  // Releases memory
			}
			neuron->b.reset();  // Releases memory
		}
	}
	layers.clear();  // Clears the layers and neurons
}
