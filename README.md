# DeepLearningTest

主要用于做一些深度学习的案例

# 案例1：用numpy进行正弦函数的近似

略



# 案例2：用torch进行正弦函数的近似

略




# 案例3：二阶常系数齐次线性微分方程的全连接神经网络和物理信息神经网络的近似

一维谐振子模型的实质是二阶常系数齐次线性微分方程，可等效为1/4悬架模型、二阶LCR电路模型

![Figure1](Figure1.png)

在不考虑重力的情况下，设质量块的质量为m，弹簧的刚度为k，阻尼器的阻尼为c，任意时刻t质量块的相对位移为y(t)，则根据牛顿第二定律有

$$
m \dfrac{d^2 y}{d t^2} + c\dfrac{d y}{d t} + ky = 0 ~
$$

设初边值条件为

$$
\left. \frac{d y}{d t} \right|_{y=0} = 0 ~
$$

$$
y(0) = 1 ~
$$


令
$
\delta = \frac{c}{2m} 
$
$
\omega_0 = \sqrt{ \frac{k}{m}} 
$

在欠阻尼的条件下

$$ \delta < \omega_0 $$

则可得其解析解为


$$ 
y(t) = 2 A e^{- \delta t} \cos{ (\omega t + \phi) } ~
$$


其中

$$
\omega = \sqrt{ \omega^2_0 + \delta^2}
$$
$$
\phi = \arctan(-\frac{\delta}{\omega})
$$
$$
A = \frac{1}{2\cos \phi} = \frac{\omega_0}{2 \sqrt{ \omega^2_0 - \delta ^2 } }
$$



设m=1 , k=400, c=4 ，忽略单位问题，则精确解为


$$
y(t) = \frac{5 \sqrt{11}}{33} e^{-2t} \cos(6 \sqrt{11} t - 0.1001674211615598)
$$

其中，三角函数为弧度制

以下将先求其解析解，然后用全连接神经网络（Fully connected neural network，FCNN）和物理信息神经网络（Physics-informed neural networks,PINN）来进行计算仿真。
