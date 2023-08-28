#include "utils.h"

// 1.clip函数
// 限制数据范围
double clip(double x, double x_min, double x_max) {
    return std::max(std::min(x, x_max), x_min);
}
// 限制向量范围
vector<double> clip(const vector<double>& x, const vector<double>& x_min, const vector<double>& x_max) {
    vector<double> result(x.size(), 0.0);
    if (x.size() != x_min.size() || x.size() != x_max.size()) {
        throw std::invalid_argument("Invalid vector dimensions.");
    }
    for (int i = 0; i < x.size(); i++) {
        result[i] = clip(x[i], x_min[i], x_max[i]);
    }
    return result;
}
vector<double> clip(const vector<double>& x, double x_min, const vector<double>& x_max) {
    return clip(x, vector<double>(x.size(), x_min), x_max);
}
vector<double> clip(const vector<double>& x, const vector<double>& x_min, double x_max) {
    return clip(x, x_min, vector<double>(x.size(), x_max));
}
vector<double> clip(const vector<double>& x, double x_min, double x_max) {
    return clip(x, vector<double>(x.size(), x_min), vector<double>(x.size(), x_max));
}

// 2.sign函数
double sign(double x) {
    return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
}
int sign(int x) {
    return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}








// 1.打印矩阵
void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


// 2.matrixTranspose函数
// 矩阵转置 A.T
vector<vector<double>> matrixTranspose(const vector<vector<double>>& matrix) {
    vector<vector<double>> result(matrix[0].size(), vector<double>(matrix.size(), 0.0));
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[0].size(); j++) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}


// 3.multiplyMatrix函数
// 矩阵乘法 A*B
vector<vector<double>> matrixMultiply(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB) {
    if (matrixA[0].size() != matrixB.size()) {
        throw std::invalid_argument("Invalid matrix dimensions.");
    }
    int m = matrixA.size();
    int n = matrixA[0].size();
    int p = matrixB[0].size();
    vector<vector<double>> result(m, vector<double>(p, 0.0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                result[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    return result;
}
// 矩阵乘法 A*b
vector<double> matrixMultiply(const vector<vector<double>>& matrixA, const vector<double>& matrixB) {
    if (matrixA[0].size() != matrixB.size()) {
        throw std::invalid_argument("Invalid matrix dimensions.");
    }
    vector<double> result(matrixA.size(), 0.0);
    for (int i = 0; i < matrixA.size(); i++) {
        for (int j = 0; j < matrixB.size(); j++) {
            result[i] += matrixA[i][j] * matrixB[j];
        }
    }
    return result;
}


// 4.matrixAddition函数
// 矩阵加法 A+B A-B
vector<vector<double>> matrixAddition(const vector<vector<double>>& matrixA, const vector<vector<double>>& matrixB, bool sub) {
    if (matrixA.size() != matrixB.size() || matrixA[0].size() != matrixB[0].size()) {
        throw std::invalid_argument("Invalid matrix dimensions.");
    }
    vector<vector<double>> result(matrixA.size(), vector<double>(matrixA[0].size(), 0.0));
    for (int i = 0; i < matrixA.size(); i++) {
        for (int j = 0; j < matrixA[0].size(); j++) {
            if (sub) {
                result[i][j] = matrixA[i][j] - matrixB[i][j];
            }
            else {
                result[i][j] = matrixA[i][j] + matrixB[i][j];
            }
        }
    }
    return result;
}
// 矩阵加法 a+b a-b
vector<double> matrixAddition(const vector<double>& matrixA, const vector<double>& matrixB, bool sub) {
    if (matrixA.size() != matrixB.size()) {
        throw std::invalid_argument("Invalid matrix dimensions.");
    }
    vector<double> result(matrixA.size(), 0.0);
    for (int i = 0; i < matrixA.size(); i++) {
        if (sub) {
            result[i] = matrixA[i] - matrixB[i];
        }
        else {
            result[i] = matrixA[i] + matrixB[i];
        }
    }
    return result;
}


// 5.matrixiIdentity函数
// 单位矩阵
vector<vector<double>> matrixiIdentity(int n) {
    vector<vector<double>> result(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        result[i][i] = 1.0;
    }
    return result;
}


// 6.matrixDeterminant函数
// 方阵行列式 |A|
double matrixDeterminant(const vector<vector<double>>& matrix) {
    if (matrix.size() != matrix[0].size()) {
        throw std::invalid_argument("Invalid matrix dimensions.");
    }

    int n = matrix.size();
    if (n == 1) {
        return matrix[0][0];
    }

    double determinant = 0.0;
    int sign = 1;
    for (int i = 0; i < n; i++) {
        vector<vector<double>> subMatrix(n - 1, vector<double>(n - 1, 0.0));
        for (int j = 1; j < n; j++) {
            int k = 0;
            for (int l = 0; l < n; l++) {
                if (l != i) {
                    subMatrix[j - 1][k++] = matrix[j][l];
                }
            }
        }
        determinant += sign * matrix[0][i] * matrixDeterminant(subMatrix);
        sign = -sign;
    }
    return determinant;
}


// 7.matrixInverse函数
// 方阵求逆 A^-1 = Adj(A) / |A|
vector<vector<double>> matrixInverse(const vector<vector<double>>& matrix) {
    if (matrix.size() != matrix[0].size()) {
        throw std::invalid_argument("Invalid matrix dimensions.");
    }

    int n = matrix.size();
    vector<vector<double>> inverse(n, vector<double>(n, 0.0));

    double determinant = matrixDeterminant(matrix);
    if (determinant == 0.0) {
        std::cout << "Matrix is not invertible." << std::endl; // 不可逆
        return inverse;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vector<vector<double>> subMatrix(n - 1, vector<double>(n - 1, 0.0));
            int p = 0;
            for (int k = 0; k < n; k++) {
                int q = 0;
                for (int l = 0; l < n; l++) {
                    if (k != i && l != j) {
                        subMatrix[p][q++] = matrix[k][l];
                    }
                }
                if (k != i) {
                    p++;
                }
            }
            inverse[j][i] = (1 / determinant) * matrixDeterminant(subMatrix) * ((i + j) % 2 == 0 ? 1 : -1);
        }
    }
    return inverse;
}


// 8.matrixPow函数
// 方阵求幂 A^n
vector<vector<double>> matrixPow(const vector<vector<double>>& matrix, int power) {
    if (matrix.size() != matrix[0].size()) {
        throw std::invalid_argument("Invalid matrix dimensions.");
    }
    
    if (power == 0) {
        return matrixiIdentity(matrix.size()); // A^0 = E
    }

    if (power <= -1) {
        return matrixPow(matrixInverse(matrix), -power); // A^(-n) = (A^-1)^n
    }
    
    vector<vector<double>> result = matrix;
    for (int i = 1; i < power; i++) {
        result = matrixMultiply(result, matrix);
    }
    return result;
}
