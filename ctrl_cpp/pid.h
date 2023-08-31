#pragma once
#include "utils.h"

//PID控制器
class PIDController {
// 控制器属性
private:
    // PID参数
    vector<double> __Kp;
    vector<double> __Ki;
    vector<double> __Kd;
    vector<pair<double, double>> __control_limits; // 控制约束
    // 先进PID参数
    vector<double> __Kf; // 前馈增益
    vector<double> __Kaw; // 抗饱和参数
    vector<pair<double, double>> __error_clip; // 积分约束
    // 记忆数据
    vector<double> __last_error; // 上一时刻误差
    vector<double> __integral_error; // 误差积分
    vector<double> __u; // 记忆控制


// 控制器设置与调用
public:
    PIDController(
        double Kp, 
        double Ki, 
        double Kd,
        const pair<double, double>& control_limits = { -Infinity, Infinity },
        double Kf = 0.0,
        double Kaw = 0.2,
        const pair<double, double>& error_clip = { -Infinity, Infinity }
    );
    /*
    :param Kp: float | list, 比例增益.
    :param Ki: float | list, 积分增益.
    :param Kd: float | list, 微分增益.
    :param control_limits: list[float] | list[list[float]], 控制量约束, 默认None表示无约束.
    :param Kf: float | list, 前馈控制增益, 默认0.
    :param Kaw: float | list, 抗积分饱和参数, 最好取0.1~0.3, 默认0.2, 取零不抗积分饱和.
    :param error_clip: float | list, list[float] | list[list[float]], 积分器分离阈值, 范围: (0, inf], 取inf时不分离积分器.
    */

    PIDController(
        const vector<double>& Kp, 
        const vector<double>& Ki, 
        const vector<double>& Kd,
        const vector<pair<double, double>>& control_limits = {},
        const vector<double>& Kf = {},
        const vector<double>& Kaw = {},
        const vector<pair<double, double>>& error_clip = {}
    );
    
    double call(double setpoint, double feedback, double dt, double expected_output = 0.0, int anti_windup_method = 1);
    /*
    :param setpoint: float | list, 参考轨迹.
    :param feedback: float | list, 系统输出.
    :param dt: float, 系统调用步长.
    :param expected_output: float | list, 期望系统输出, 默认没有.
    :param anti_windup_method: Literal[0, 1], 抗积分饱和方法, 默认1.
    */
    double operator()(double setpoint, double feedback, double dt, double expected_output = 0.0, int anti_windup_method = 1);
    vector<double> call(const vector<double>& setpoint, const vector<double>& feedback, double dt, const vector<double>& expected_output = {}, int anti_windup_method = 1);
    vector<double> operator()(const vector<double>& setpoint, const vector<double>& feedback, double dt, const vector<double>& expected_output = {}, int anti_windup_method = 1);


// @classmethod
private:
    static double _AntiWindup(double Ki, double& ierror, double error, pair<double, double> error_lim, double u, pair<double, double> u_lim, double Kaw, double dt, int method = 1); // 返回积分项
    /* integral_error是误差积分, integral是积分项, 返回积分项, 同时记忆总误差积分 */
};
