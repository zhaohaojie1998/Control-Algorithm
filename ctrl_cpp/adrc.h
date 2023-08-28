#pragma once
#include "utils.h"

// ADRC控制器
class ADRCController
{
// 控制器属性
private:
	// ADRC参数
	vector<double> __r;
	vector<double> __b0;
	vector<double> __delta;
	vector<double> __beta01;
	vector<double> __beta02;
	vector<double> __beta03;
	vector<double> __alpha1;
	vector<double> __alpha2;
	vector<double> __beta1;
	vector<double> __beta2;
	vector<pair<double, double>> __control_limits;
	// 记忆数据
	vector<double> __v1;
	vector<double> __v2;
	vector<double> __z1;
	vector<double> __z2;
	vector<double> __z3;
	vector<double> __u;


// 一维控制器
public: 
	ADRCController(
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
		const pair<double, double>& control_limits = {}
	);

	double call(double setpoint, double feedback, double dt, int ctrl_low = 0);
	double operator()(double setpoint, double feedback, double dt, int ctrl_low = 0);


// 多维控制器	
	ADRCController(
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
		const vector<pair<double, double>>& control_limits = {}
	);

	vector<double> call(const vector<double>& setpoint, const vector<double>& feedback, double dt, int ctrl_low = 0);
	vector<double> operator()(const vector<double>& setpoint, const vector<double>& feedback, double dt, int ctrl_low = 0);


// @staticmethod
private: 
	// 1.跟踪微分器 v1, v2 <- TD(v, v1, v2, r, h)
	static void _td(
		double v, 
		double& v1, double& v2, 
		double r, double h
	); 
	// 2.扩张状态观测器 z1, z2, z3 <- ESO(y, z1, z2, z3, b0, u, beta01, beta02, beta03, delta, h)
	static void _eso(
		double y, 
		double& z1, double& z2, double& z3, 
		double b0, double u,
		double beta01, double beta02, double beta03, 
		double delta, double h
	); 
	// 3.非线性状态误差反馈控制律 
	// u0 = nlsef(e1, e2, alpha1, alpha2, beta1, beta2, delta)
	static double _nlsef(
		double e1, double e2,
		double alpha1, double alpha2,
		double beta1, double beta2,
		double delta
	); 
	// u0 = _nlsef(e1, e2, r, h, c)
	static double _nlsef(
		double e1, double e2,
		double r, double h, double c // 这里C不能有默认参数, 会导致函数重载失败
	); 
	// u0 = nlsef(e1, e2, beta1, beta2) 线性反馈
	static double _nlsef(
		double e1, double e2,
		double beta1, double beta2
	);
	// 4.非线性函数
	static double _fhan(double x1, double x2, double r, double h); // fh = fhan(x1, x2, r, h)
	static double _fal(double e, double alpha, double delta); // fa = fal(e, alpha, delta)
	static double _fsg(double x, double d); // sign(x + d) - sign(x - d)) / 2
};

