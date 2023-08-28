#include "adrc.h"

// @staticmethod
// 1.跟踪微分器 v1, v2 <- TD(v, v1, v2, r, h)
void ADRCController::_td(double v, double& v1, double& v2, double r, double h) {
    double fh = _fhan(v1 - v, v2, r, h); // 用老的数据计算
    v1 += h * v2;
    v2 += h * fh;
}

// 2.扩张状态观测器 z1, z2, z3 <- ESO(y, z1, z2, z3, b0, u, beta01, beta02, beta03, delta, h)
void ADRCController::_eso(double y, double& z1, double& z2, double& z3, double b0, double u, double beta01, double beta02, double beta03, double delta, double h) {
    double e = z1 - y;
    double fe = _fal(e, 1.0 / 2, delta);
    double fe1 = _fal(e, 1.0 / 4, delta);
    z1 += h * (z2 - beta01 * e);
    z2 += h * (z3 - beta02 * fe + b0 * u);
    z3 += h * (-beta03 * fe1);
    // todo: clip(z1, -10000, 10000)
}

// 3.非线性状态误差反馈控制律 
// u0 = nlsef(e1, e2, alpha1, alpha2, beta1, beta2, delta)
double ADRCController::_nlsef(double e1, double e2, double alpha1, double alpha2, double beta1, double beta2, double delta) {
    return beta1 * _fal(e1, alpha1, delta) + beta2 * _fal(e2, alpha2, delta);
}
// u0 = _nlsef(e1, e2, r, h, c=1)
double ADRCController::_nlsef(double e1, double e2, double r, double h, double c) {
    return - _fhan(e1, c * e2, r, h);
}
// u0 = nlsef(e1, e2, beta1, beta2) 线性反馈
double ADRCController::_nlsef(double e1, double e2, double beta1, double beta2) {
    return beta1 * e1 + beta2 * e2;
}

// 4.非线性函数
double ADRCController::_fhan(double x1, double x2, double r, double h) {
    double d = r * pow(h, 2);
    double a0 = h * x2;
    double y = x1 + a0;
    double a1 = sqrt(d * (d + 8 * abs(y)) + 1e-8);
    double a2 = a0 + sign(y) * (a1 - d) / 2;
    double a = (a0 + y) * _fsg(y, d) + a2 * (1 - _fsg(y, d));
    return -r * (a / d) * _fsg(y, d) - r * sign(a) * (1 - _fsg(a, d));
}
double ADRCController::_fal(double e, double alpha, double delta) {
	return (abs(e) <= delta) ? e / pow(delta, alpha - 1) : sign(e) * pow(abs(e), alpha);
}
double ADRCController::_fsg(double x, double d) {
	return (sign(x + d) - sign(x - d)) / 2;
}



// init adrc
ADRCController::ADRCController(
    double r,
    double b0,
    double delta,
    double eso_beta01,
    double eso_beta02,
    double eso_beta03,
    double nlsef_alpha1,
    double nlsef_alpha2,
    double nlsef_beta1,
    double nlsef_beta2,
    const pair<double, double>& control_limits
) :
    __r(1, r),
    __b0(1, b0),
    __delta(1, delta),
    __beta01(1, eso_beta01),
    __beta02(1, eso_beta02),
    __beta03(1, eso_beta03),
    __alpha1(1, nlsef_alpha1),
    __alpha2(1, nlsef_alpha2),
    __beta1(1, nlsef_beta1),
    __beta2(1, nlsef_beta2),
    __control_limits(1, control_limits),
    __v1(1, 0.0),
    __v2(1, 0.0),
    __z1(1, 0.0),
    __z2(1, 0.0),
    __z3(1, 0.0),
    __u(1, 0.0)
{
    if (nlsef_alpha1 >= 1 || nlsef_alpha1 <= 0) {
        throw std::invalid_argument("0 < alpha1 < 1");
    }
    if (nlsef_alpha2 <= 1) {
        throw std::invalid_argument("alpha2 > 1");
    }

}

ADRCController::ADRCController(
    const vector<double>& r,
    const vector<double>& b0,
    const vector<double>& delta,
    const vector<double>& eso_beta01,
    const vector<double>& eso_beta02,
    const vector<double>& eso_beta03,
    const vector<double>& nlsef_alpha1,
    const vector<double>& nlsef_alpha2,
    const vector<double>& nlsef_beta1,
    const vector<double>& nlsef_beta2,
    const vector<pair<double, double>>& control_limits
):
    __r(r),
    __b0(b0),
    __delta(delta),
    __beta01(eso_beta01),
    __beta02(eso_beta02),
    __beta03(eso_beta03),
    __alpha1(nlsef_alpha1),
    __alpha2(nlsef_alpha2),
    __beta1(nlsef_beta1),
    __beta2(nlsef_beta2),
    __control_limits(control_limits),
    __v1(nlsef_beta1.size(), 0.0),
    __v2(nlsef_beta1.size(), 0.0),
    __z1(nlsef_beta1.size(), 0.0),
    __z2(nlsef_beta1.size(), 0.0),
    __z3(nlsef_beta1.size(), 0.0),
    __u(nlsef_beta1.size(), 0.0)
{}



// call adrc
double ADRCController::call(double setpoint, double feedback, double dt, int ctrl_low) {
    return this->call(vector<double>{setpoint}, vector<double>{feedback}, dt, ctrl_low)[0];
}
double ADRCController::operator()(double setpoint, double feedback, double dt, int ctrl_low) {
    return this->call(setpoint, feedback, dt, ctrl_low);
}

vector<double> ADRCController::call(const vector<double>& setpoint, const vector<double>& feedback, double dt, int ctrl_low) {
    // 检查维度
    if (setpoint.size() != feedback.size() || setpoint.size() != __r.size()) {
        throw std::invalid_argument("Dimension mismatch between setpoint, feedback, and parameters.");
    }
    ctrl_low = ctrl_low % 3;
    for (int i = 0; i < __r.size(); i++) {
        // 跟踪微分器 TD
        _td(setpoint[i], __v1[i], __v2[i], __r[i], dt);
        // 扩张状态观测器 ESO
        _eso(feedback[i], __z1[i], __z2[i], __z3[i], __b0[i], __u[i], __beta01[i], __beta02[i], __beta03[i], __delta[i], dt);
        // 非线性状态误差反馈控制律 NLSEF
        double e1 = __v1[i] - __z1[i];
        double e2 = __v2[i] - __z2[i];
        double u0;
        if (ctrl_low == 0) {
            u0 = _nlsef(e1, e2, __alpha1[i], __alpha2[i], __beta1[i], __beta2[i], __delta[i]);
        }
        else if (ctrl_low == 1) {
            double c = 1.5; // 这个参数不外露了
            u0 = _nlsef(e1, e2, __r[i], dt, c);
        }
        else {
            u0 = _nlsef(e1, e2, __beta1[i], __beta2[i]);
        }
        // 更新控制律
        __u[i] = u0 - __z3[i] / __b0[i];
        
        if (!__control_limits.empty() && i < __control_limits.size()) {
            __u[i] = clip(__u[i], __control_limits[i].first, __control_limits[i].second);
        }
    }
    return __u;
}
vector<double> ADRCController::operator()(const vector<double>& setpoint, const vector<double>& feedback, double dt, int ctrl_low) {
    return this->call(setpoint, feedback, dt, ctrl_low);
}