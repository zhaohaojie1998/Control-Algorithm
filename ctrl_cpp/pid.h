#pragma once
#include "utils.h"

//PID控制器
class PIDController {
public:
    PIDController(
        double Kp, 
        double Ki, 
        double Kd,
        pair<double, double> control_limits = {}, // NOTE 默认参数只能设置一次, 头文件设置了源文件就不要有了
        pair<double, double> integral_limits = {},
        double derivative_filter_coef = 0.0,
        double feedforward_gain = 0.0
        );

    double call(double setpoint, double feedback, double dt, double expected_output = 0.0);
    double operator()(double setpoint, double feedback, double dt, double expected_output = 0.0);

    PIDController(
        const vector<double>& Kp, 
        const vector<double>& Ki, 
        const vector<double>& Kd,
        const vector<pair<double, double>>& control_limits = {},
        const vector<pair<double, double>>& integral_limits = {},
        const vector<double>& derivative_filter_coef = {},
        const vector<double>& feedforward_gain = {}
    );
    
    vector<double> call(const vector<double>& setpoint, const vector<double>& feedback, double dt, const vector<double>& expected_output = {});
    vector<double> operator()(const vector<double>& setpoint, const vector<double>& feedback, double dt, const vector<double>& expected_output = {});


private:
    vector<double> __Kp;
    vector<double> __Ki;
    vector<double> __Kd;
    vector<double> __last_error;
    vector<double> __integral;

    vector<pair<double, double>> __control_limits; // 控制约束
    vector<pair<double, double>> __integral_limits; // 积分约束
    vector<double> __derivative_filter_coef; // 微分滤波
    vector<double> __feedforward_gain; // 前馈增益
};
