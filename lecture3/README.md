### CS224n Lecture3

#### Classifcation

Classification(분류) 문제는 어떤 data쌍 ($x_i, y_i$)가 있을 때, input feature($x_i$)가 주어지면 이것이 어떤 class($y_i$)에 속하는지 예측하는 문제이다. NLP에서 input data는 주로 단어들, 문장들, 도큐먼트들 등이 될 수 있고, class는 sentiment, named entity등이 될 수 있다. 이때, 모델은 특정 차원의 input vector space에서의 decision boundary(hyperplane)을 학습하고자 한다.

#### Traditional ML/Stats approach

- Method: Matrix multiplication($Wx+b$)를 통해 feature를 linear transform하고 softmax/logistic function등의 함수를 통과하여 얻은 output($p(x|y)$)를 최대화하는 모델(decision boundary)을 학습한다. - single layer

- "Cross Entropy"

  - information theory에서 두 probability distribution사이의 차이를 계산하는 방법
  - $H(p,q) = -p(c)log(q(c))​$
  - classification model에서는 정답 label(y)의 분포(p(c))와 model이 예측한 분포(q(c))의 차이를 최소화해야한다.

- Logistic regression

  - target distribution이(y의 분포) 베르누이 분포인 경우 이용가능 [0, 1] - binary classification
  - Matrix multiplication($Wx+b$)를 통해  **"1개"**의 output을 얻고, logistic function을 이용하여 일종의 확률 점수를 얻는다.
  - $logistic\space function\space L(x) = \frac{1} {1+e^{-x}}​$ 
  - sigmoid function을 통과하면 모든 값이 0~1사이의 값으로 mapping됨 - 0.5를 기준으로 참/거짓(logistic)

  - $H(p,q) =  -p(c)log(q(c)) - (1-p(c))log(q(c))$

  - p(c) = [0] or [1] / q(c)는 logistic function을 통과한 분포

- Softmax Classifier

  - target distribution이(y의 분포) 다항 분포인 경우 이용가능 - multi class classification
  - Matrix multiplication($Wx+b$)를 통해 **"class 개수"** 만큼의 output을 얻고, softmax function을 이용하여 각 class의 확률분포 계산

  		$p(y|x) = \frac{exp(W_yx)}{\Sigma_{c=1}^C exp(W_cx)}$

  - Softmax 함수를 적용하면 특정 값들을 normalized probability로 변환할 수 있다.
    - data x가 주어졌을 때, True label y(정답 class)가 나올 조건부 확률 p(y|x)를 계산할 수 있다.
  -  우리는 학습을 통해 p(y|x)를 최대화 하고자 한다. - Maximum Likelihood Estimation
    - $Max(p(y|x)) = Min(-log(p(y|x)))​$ - minimize negative log probability
  - $H(p,q) =  \sum_{c=1}^{C}-p(c)log(q(c))​$
  - p(c) = [0,0…1(정답),…0,0]의 분포/ q(c)는 softmax를 통과한 normalized probability분포
  - 정답외의 모든 p(c)분포가 0 - 결과적으로 classification에서 cross entropy를 계산하면, negative log probability와 같아진다. 

- 한계

  - 전통적인 ML의 softmax(logistic regression)모델은 linear decision boundary만 학습할 수 있다.
  - 복잡한 문제를 풀기에 적합하지 않다 - high bias problem
  - Neural classifier를 쓰자!

#### Neural Classifier

- Neuron의 작용을 흉내내는것에서 시작 - 뉴런은 binary logistic regression unit일 수 있다!
- Neural network: 각각 logistic regression을 수행하는 몇몇 unit(neuron)을 동시에 실행시키는(한 layer의 unit들을 동시에 계산) 네트워크
- 여러개의 logistic regression unit을 포함하는 layer들을 거쳐서 결과를 얻음 - multi layer
- $h_{w,b}=f(w^Tx+b), f(z) =\frac{1}{1+e^{-z}}$ - f는 logistic fuction, x는 이전 layer의 결과
- 전통적인 ML과 차이점
  - Neural net은 각 unit이 f(z) (logistic function = "Non-Linear" function)을 거치기 때문에 "Non-linear"한 decision boundary를 학습할 수 있다.
  - Non linearity를 통과하기 때문에 layer를 여러개 쌓을 수 있다. - Linear한 layer를 여러개 쌓아도 결과적으로 linear transform일뿐이다.

#### Classification 예시 - NER(Named Entity Recognition)

NER(Named Entity Recognition)은 location, person, organization등 특정 이름들을 분류하는 문제이다. 단어 하나만 보면 어려운 문제(다의어, 애매함)이기 때문에 특정 window 내부의 context word들을 보고 center word의 entitiy를 맞추는 문제로 접근한다.

- Method1) 특정 window내의 단어들의 representation을 평균하여 classify - 단어의 순서가 고려되지 않음
- Method2) 단어들의 representation을 concat하여 사용 - 단어의 순서 보존 but parameter가 많아짐

#### Computing Gradient

Stochastic gradient descent algorithm에서 gradient의 반대방향으로 $\alpha​$(learning rate) 만큼 파라메터를 학습시킨다.

$\theta^{new} = \theta^{old} - \alpha \nabla_{\theta}J(\theta)$

$\nabla_{\theta}J(\theta) = [\frac{\partial J}{\partial \theta_1}, \frac{\partial J}{\partial \theta_2}, \frac{\partial J}{\partial \theta_3}, ... \frac{\partial J}{\partial \theta_n}]$ : **gradient** - objective function $J$ 를 $\theta=[\theta_1,\theta_2,\theta_3,... \theta_n]$ 로 미분한 값

- $y = f(x)$ , $f: R^{1} \rightarrow R^{1}$ 이면 미분 값은 $\frac{dy}{dx}$ 
- $y=f(x_1, x_2, x_3, ..., x_n)$ , $f : R^n \rightarrow R^1$ 이면 미분 값(gradient)은 $\nabla_x f = [\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \frac{\partial f}{\partial x_3},...\frac{\partial f}{\partial x_n}]$
- $y_1, y_2, y_3, ... y_m = [f_1(x_1, x_2, x_3, ... x_n), f_2(x_1, x_2, x_3, ... x_n), f_3(x_1, x_2, x_3, ... x_n), ... f_m(x_1, x_2, x_3, ... x_n)]$ , $f:R^n \rightarrow R^m$ 이면 미분 값(Jacobian) $\frac {\partial f}{\partial x} = \begin {bmatrix} \frac {\partial f_1}{\partial x_1}  & ... & \frac {\partial f_1}{\partial x_n}\\...&&...\\ \frac {\partial f_m}{\partial x_1} &...& \frac {\partial f_1}{\partial x_n} \end{bmatrix}$ 

Chain rule: 합성 함수의 미분 - $z = f(y), y=g(x)$ 일 때, $ \frac{dz}{dx} =\frac{dz}{dy} \frac{dy}{dx} $ 

