#pragma once
#include <iostream>
#include <algorithm> // for std::min, std::max
#include <stdexcept> // for std::invalid_argument
#include <vector> // for std::vector
//#include <Eigen/Core> // matlab
using std::vector;
using std::pair;

constexpr auto Infinity = 100000000.0;


// 限制数据范围 <cmath>里有std::clamp
double clip(double x, double x_min, double x_max);
vector<double> clip(const vector<double>& x, const vector<double>& x_min, const vector<double>& x_max);
vector<double> clip(const vector<double>& x, double x_min, const vector<double>& x_max);
vector<double> clip(const vector<double>& x, const vector<double>& x_min, double x_max);
vector<double> clip(const vector<double>& x, double x_min, double x_max);

// 符号函数
double sign(double x);
int sign(int x);

// 打印矩阵
void printMatrix(const vector<vector<double>>& matrix);

// 矩阵转置
vector<vector<double>> matrixTranspose(const vector<vector<double>>& matrix);

// 矩阵乘法
vector<vector<double>> matrixMultiply(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB);
vector<double> matrixMultiply(const vector<vector<double>>& matrixA, const vector<double>& matrixB);

// 矩阵加法
vector<vector<double>> matrixAddition(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB, bool sub = false);
vector<double> matrixAddition(const vector<double>& matrixA, const vector<double>& matrixB, bool sub = false);

// 单位矩阵
vector<vector<double>> matrixiIdentity(size_t n);

// 方阵行列式
double matrixDeterminant(const vector<vector<double>>& matrix);

// 方阵求逆
vector<vector<double>> matrixInverse(const vector<vector<double>>& matrix);

// 方阵求幂
vector<vector<double>> matrixPow(const vector<vector<double>>& matrix, int power);

// 矩阵求秩