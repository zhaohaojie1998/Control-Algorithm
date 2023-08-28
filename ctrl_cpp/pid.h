#pragma once
#include "utils.h"

//PID控制器
class PIDController {
private:
    vector<double> __Kp;
    vector<double> __Ki;
    vector<double> __Kd;
    vector<double> __last_error;
    vector<double> __integral;

    vector<pair<double, double>> __control_limits; // 控制约束
    vector<pair<double, double>> __integral_limits; // 积分约束
    vector<double> __feedforward_gain; // 前馈增益

public:
    PIDController(
        double Kp, 
        double Ki, 
        double Kd,
        const pair<double, double>& control_limits = {}, // NOTE 默认参数只能设置一次, 头文件设置了源文件就不要有了
        const pair<double, double>& integral_limits = {},
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
        const vector<double>& feedforward_gain = {}
    );
    
    vector<double> call(const vector<double>& setpoint, const vector<double>& feedback, double dt, const vector<double>& expected_output = {});
    vector<double> operator()(const vector<double>& setpoint, const vector<double>& feedback, double dt, const vector<double>& expected_output = {});

};
