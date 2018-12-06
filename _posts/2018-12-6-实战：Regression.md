---
tag: 机器学习 机器学习实战
---



## 实战：regression

1. 概述：通过前n个小时的各种数据（例如空气中各种污染物含量、降雨量、风向风速等）来预测下一个小时的PM2.5。[链接](https://ntumlta.github.io/2017fall-ml-hw1/)
2. 步骤提要：
   1. Training Data 的数据分析与收集
   2. 建立模型
   3. 定义损失函数
   4. 通过 Adagrad进行训练，完善模型
   5. 在Test data 上进行测试



3. 具体代码：





```
#include<iostream>
#include<fstream>
#include<string>
#include<cstring>
#include<cmath>
using namespace std;
double training_total_data[12][20][24];  //总的数据
double  training_result[3600];            //机器算出的结果
double  training_correction[3600];	    // 正确的结果（y^hat)
double b = 2;                                //bias
double w[9] = { 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };							//9个参数
#define learning_rate 1E-2
double eachpartial[10][3600 * 1000];

double model(double data[][20][24], int m, int d, int h)       //线性函数模型
{
	double sum = b;
	for (int iw = 0; iw < 9; iw++)
		sum = sum + w[iw] * data[m][d][h + iw];
	return sum;

}
//定义训练集上的损失函数: L= mean square 测试集合共有 15*20*12 数据（每天15个，20天，12个月）
double loss(double tr[], double tc[])
{
	double result = 0;
	for (int i = 0; i < 3600; i++)
		result = result + (tr[i] - tc[i])*(tr[i] - tc[i]);
	return result / 3600;                    

}


double retpartial(double yhat[], double training_result[], double tr_data[][20][24], int para)    //返回一个偏微分
{
	if (para == -1)
	{
		double result = 0;
		int i = 0;
		for (int m = 0; m < 12; m++)
		for (int d = 0; d < 20; d++)
		{
			for (int h = 0; h < 15; h++)
			{
				result = result + 2 * (yhat[i] - training_result[i])*(-1);
				i++;
			}
		}
		return result;
	}
	double result = 0;
	int i = 0;
	for (int m = 0; m < 12; m++)
	for (int d = 0; d < 20; d++)
	{
		for (int h = 0; h < 15; h++)
		{
			result = result + 2 * (yhat[i] - training_result[i])*(-1)* tr_data[m][d][h + para];
			i++;
		}
	}
	return result;

}

double retroot(int para, int j, double eachpartial[][3600 * 1000])     //Adagrad 里面的分母项
{
	double sum = 0;
	for (int i = 0; i <= j; i++)
		sum = sum + eachpartial[para][i] * eachpartial[para][i];
	return sqrt(sum);

}
int main()
{
	//1.收集数据到开头的数组里面
	ifstream in;
	in.open("train.txt");
	string tmpb;
	int m = 0;
	int d = 0;
	int h = 0;
	while (true)
	{
		in >> tmpb;
		if (tmpb == "结束")
		{
			in.close();
			break;
		}
		while (tmpb == "PM2.5")
		{
			if (d == 20)
			{
				d = 0;
				m++;
			}
			for (h = 0; h < 24; h++)
			{
				in >> training_total_data[m][d][h];

			}
			d++;
			break;
		}
	}
	//来测试一下数据对不对
	int testi = 1;
	for (int m = 0; m < 12; m++)
	{
		for (int d = 0; d < 20; d++)
		{
			cout << testi << ":    ";
			testi++;
			for (int h = 0; h < 24; h++)
			{
				cout << training_total_data[m][d][h] << ' ';
			}
			cout << endl;
		}
	}
	system("pause");
	// 采集正确y^hat
	{
		int i = 0;
		for (int m = 0; m < 12; m++)
		for (int d = 0; d < 20; d++)
		for (int h = 9; h < 24; h++)
		{
			training_correction[i] = training_total_data[m][d][h];
			i++;
		}
	}
	//开始训练
	int times = 0;
	int actforpartial = 1;
	while (true)
	{
		//计算结果
		int i = 0;
		for (int m = 0; m < 12; m++)
		for (int d = 0; d < 20; d++)
		for (int h = 0; h < 15; h++)
		{
			training_result[i] = model(training_total_data, m, d, h);
			i++;
		}
		//更新参数
		for (int para = 0; para < 9; para++)
		{
			eachpartial[para][actforpartial] = retpartial(training_correction, training_result, training_total_data, para);

			w[para] = w[para] - (learning_rate / retroot(para, actforpartial, eachpartial))*eachpartial[para][actforpartial];
		}
		//更新bias
		eachpartial[9][actforpartial] = retpartial(training_correction, training_result, training_total_data, -1);
		b = b - (learning_rate / retroot(9, actforpartial, eachpartial))*eachpartial[9][actforpartial];

		actforpartial++;
		//每隔500轮训练输出一次结果
		if (times % 500 == 0)
		{
			cout << "Loss Value:  " << loss(training_result, training_correction) << endl;
			for (int t = 0; t < 9; t++)
				cout << w[t] << endl;
			cout << b << endl;
		}
		if (times == 300000)
			break;
		times++;
	}

	system("pause");
	return 0;
}
```







### 结果分析

1. 在设计模型的时候，首先考虑到空气中其他污染物浓度和PM2.5 之间有很强的关联性（行列式的值要么因为成比例为0要么是多余的），所以这里只取用了PM2.5 一项作为训练Data。
2. 在数据分析手段不是很充分的条件下，没有考虑到风向、风速、降水或者地形地势等条件，简化了模型。
3. 采用的 Linear-Regression， 通过前9个小时的PM2.5 数据来预测接下来1个小时的PM2.5的值。





![Result](https://i.loli.net/2018/12/06/5c0902bea9788.png)

4. 如图所示，最终在训练集上Loss Value(计算值和真实值的差的平方的平均数)为43.1495,也就是说平均每一个PM2.5 相差6.56 左右。
5. 因为天气系统的复杂性，只取用了PM2.5的数据（极少量数据）来预测，这个误差相对较大但也在情理之中。当然也可能是代码写错或者是C++ 自身的原因。











