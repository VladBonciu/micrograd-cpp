#pragma once

#include "Value.h"
#include <iostream>
#include <vector>
#include <memory>
#include <ctime>
#include <random>

class Neuron : public std::enable_shared_from_this<Neuron>
{
public:
	Neuron();
	Neuron(int n_in, bool _apply_activation);

	std::shared_ptr<Value> call(std::vector<std::shared_ptr<Value>>& x);

	std::vector<std::shared_ptr<Value>> w;
	std::shared_ptr<Value> b;
	bool apply_activation;
};

class Layer : public std::enable_shared_from_this<Layer>
{
public:
	Layer();
	Layer(int n_in, int n_out, bool _apply_activation);

	std::vector<std::shared_ptr<Value>> call(std::vector<std::shared_ptr<Value>>& x);

	std::vector<std::shared_ptr<Neuron>> neurons;
};

class MLP
{
public:
	MLP();
	MLP(std::vector<int>& layers);

	std::vector<std::shared_ptr<Value>> call(std::vector<std::shared_ptr<Value>>& x);
	std::vector<std::shared_ptr<Value>> get_params();

	void reset();

	std::vector<std::shared_ptr<Layer>> layers;
};



