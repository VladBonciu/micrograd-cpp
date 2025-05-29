#include "BatchReadCSV.h"

std::vector<std::vector<std::string>> read_csv(const char* file_path)
{
    std::fstream fin;
    fin.open(file_path, std::ios::in);


    if (!fin.is_open())
    {
        std::cout << "Error: file not opened!";
        return std::vector<std::vector<std::string>>(0);
    }

    int row_count = 0;
    int col_count = 0;

    std::vector<std::vector<std::string>> out;

    std::string line, word, temp;

    while (getline(fin, line))
    {
        std::vector<std::string> row;

        row.clear();

        // used for breaking words
        std::stringstream s(line);

        row_count++;

        while (std::getline(s, word, ','))
        {
            // add all the column data
            // of a row to a vector
            row.push_back(word);
        }

        out.push_back(row);
    }

    fin.close();

    std::cout << "Succesfully read .csv data.\n";

    return out;
}

int read_csv_input_count(const char* file_path)
{
    std::fstream fin;
    fin.open(file_path, std::ios::in);


    if (!fin.is_open())
    {
        std::cout << "Error: file not opened!";
        return -1;
    }

    int row_count = 0;

    std::string line, word, temp;

    while (getline(fin, line))
    {
        row_count++;
    }

    fin.close();

    row_count--;

    std::cout << "Elements in the .csv file:" << row_count << "\n";

    return row_count;
}

std::vector<std::vector<std::string>> read_csv_batch(const char* file_path)
{
    std::fstream fin;
    fin.open(file_path, std::ios::in);


    if (!fin.is_open())
    {
        std::cout << "Error: file not opened!";
        return std::vector<std::vector<std::string>>(0);
    }

    int row_count = 0;
    int col_count = 0;

    std::vector<std::vector<std::string>> out;

    std::string line, word, temp;

    while (getline(fin, line))
    {
        std::vector<std::string> row;

        row.clear();

        // used for breaking words
        std::stringstream s(line);

        row_count++;

        while (std::getline(s, word, ','))
        {
            // add all the column data
            // of a row to a vector
            row.push_back(word);
        }

        out.push_back(row);
    }

    fin.close();

    std::cout << "Succesfully read .csv data.\n";

    return out;
}

std::vector<std::vector<std::string>> shuffle_data(std::vector<std::vector<std::string>> in)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1, in.size()-1);

    for (int i = 1; i < in.size(); i++)
    {   
        int a = dist(rng);
        int b = dist(rng);
        std::vector<std::string > aux = in[a];
        in[a] = in[b];
        in[b] = aux;
    }

    std::cout << "Succesfully shuffed string data.\n";

    return in;
}

void split_data_80_20(std::vector<std::vector<std::string>> in, std::vector<std::vector<std::string>>& train, std::vector<std::vector<std::string>>& test)
{
    int train_size = in.size() * 0.8;
    int test_size = in.size() - train_size;

    train.clear();
    test.clear();

    train = std::vector<std::vector<std::string>>(train_size);
    test = std::vector<std::vector<std::string>>(test_size);

    for (int i = 0; i < train_size; i++)
    {
        train[i] = std::vector<std::string>(in[i].size());
        train[i].assign(in[i].begin(), in[i].end());
    }

    for (int i = train_size; i < in.size(); i++)
    {
        int j = i - train_size;
        test[j] = std::vector<std::string>(in[i].size());
        test[j].assign(in[i].begin(), in[i].end());
    }

    std::cout << "Succesfully split the string data. (80-train/20-test)\n";
}


std::vector<std::vector<double>> convert_string_to_double_matrix(std::vector<std::vector<std::string>> in)
{
    std::vector<std::vector<double>> out = std::vector<std::vector<double>>(in.size() - 1);
    for (int i = 0; i < (in.size() - 1);i++)
    {
        out[i] = std::vector<double>(in[i + 1].size());
    }

    for (int i = 1; i < in.size(); i++)
    {
        for (int j = 0; j < in[i].size()-1; j++)
        {
            //std::cout << in[i][j] << " ";
            out[i - 1][j] = std::stod(in[i][j]);
        }

        if (in[i][in[i].size() - 1] == "Iris-setosa")
            out[i - 1][in[i].size() - 1] = 0;
        else if(in[i][in[i].size() - 1] == "Iris-versicolor")
            out[i - 1][in[i].size() - 1] = 1;
        else
            out[i - 1][in[i].size() - 1] = 2;

        //std::cout << out[i-1][in[i].size() - 1] << " ";
        //std::cout <<  "\n";
    }

    std::cout << "Succesfully converted string data to double.\n";

    return out;
}

//int main(void)
//{
//    std::vector<std::vector<std::string>> data = read_csv("mnist_test.csv");
//    std::vector<std::vector<double>> p_data = convert_string_to_double_matrix(data);
//    
//    for (int i = 0; i < p_data.size();i++)
//    {
//        std::cout << p_data[i][0] << "\n";
//    }
//
//    std::cout << p_data.size() << "\n";
//
//    return 0;
//}