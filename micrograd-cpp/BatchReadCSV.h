#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <random>

std::vector<std::vector<std::string>> read_csv(const char* file_path);
std::vector<std::vector<std::string>> shuffle_data(std::vector<std::vector<std::string>> in);
std::vector<std::vector<double>> convert_string_to_double_matrix(std::vector<std::vector<std::string>> in);
void split_data_80_20(std::vector<std::vector<std::string>> in, std::vector<std::vector<std::string>>& train, std::vector<std::vector<std::string>>& test);
int read_csv_input_count(const char* file_path);