# adaboost-face-detection

Implement [Rapid Object Detection Using a Boosted Cascade of Simple Features](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)

### 1. Principle part

#### 1.1. The basic idea of $AdaBoost$

The full name of $Adaboost$ is $Adaptive\ Boosting$, which is an adaptive boosting method, which can be used to solve classification problems. Its connotation is to first establish a weak classifier, and after classifying the samples, it will increase the weight of those misclassified samples, reduce the weight of those correctly classified samples, and iterate this process continuously. Each round of iteration is a weak classifier. In this way, those data that have not been correctly classified will receive more attention from the weak classifiers in the next round due to the increase in their weights. This is a round of promotion. Finally, when the error is less than a threshold, all weak classifiers can be combined into a strong classifier, which is $AdaBoost$.

> Weak classifier: If there is a polynomial learning algorithm, so that the correct rate of learning is only slightly better than random guessing, for example, around $50\%$~$60\%$, then this algorithm is called weakly learnable, which is a weak classifier.

#### 1.2. Algorithm process of $AdaBoost$

Existing training data set $T=\{(x_1,y_1),(x_2,y_2),...,(x_n,y_n)\}$, where each sample point consists of an instance and a label, instance $x_i\in\chi\subseteq\mathcal{R}^n$, label $y_i\in\mathcal{Y}=\{-1,+1\}$, $\chi $ is the instance space, $\mathcal{Y}$ is the set of markers, we want to output a final classifier function $G(x)$

(1) Initialize the weight distribution of the training data:

$D_1=(w_{11},w_{12},...,w_{1i},...,w_{1N})$, $w_{1i}=\frac{1}{N}$, $i=1,2,...,N$

Even if it is assumed that in the first round of weak classifiers, each training sample has the same weight, it is fair

(2) For $m=1,2,...,M$, it indicates the number of iterations

 (a) Use the training data set with weight distribution $D_m$ to learn, and get the basic classifier: $G_m(x):\chi\rightarrow\{-1,+1\}$

 (b) Calculate the classification error rate of $G_m(x)$ on the training data set $e_m=\sum_{i=1}^NP(G_m(x_i)\neq y_i)=\sum_{i=1}^Nw_{mi}I_{\{G_m(x_i)\neq y_i\}}$

 (c) Calculate the coefficient of $G_m(x)$: $\alpha_m=\frac{1}{2}ln\frac{1-e_m}{e_m}$

 (d) Update the weight distribution of the training data set: $D_{m+1}=(w_{m+1,1},w_{m+1,2},...,w_{m+1,i},...,w_{m+1,N})$

 where $w_{m+1,i}=\frac{w_{mi}\cdot e^{-\alpha_my_iG_m(x_i)}}{\sum_{i=1}^Nw_{mi}\cdot e^{-\alpha_my_iG_m(x_i)}}$, $i=1,2,...,N$

(3) Construct a linear combination of basic classifiers: $f(x)=\sum_{m=1}^M\alpha_mG_m(x)$, and get the final classifier $G(x)=sign\big(f(x)\big)=sign\big(\sum_{m=1}^M\alpha_mG_m(x)\big)$

#### 1.3. Algorithm derivation of $AdaBoost$

##### 1.3.1 What is the loss function in the above algorithm?

There are many things in the above algorithm process, such as how the coefficient of $G_m(x)$ comes from, where is the loss function, how to minimize the loss function, and how to ensure that the algorithm is convergent, why the weights need to be updated in this way, will now be explained here one by one.

A certain round of classifier is $f_m(x)=f_{m-1}(x)+\alpha_mG_m(x)$, in addition, $AdaBoost$ uses an exponential loss function, and the loss function form of each round is $L(y,f(x))=\frac{1}{N}\sum_{i=1}^N e^{-y_if(x_i)}$, where $y_i$ is the real value, $f( x_i)$ is a prediction function with weight, greater than $0$ represents the positive class, and it may be less than $0$ represents the negative class, $y_i=\{-1,+1\}$, it can be seen that if the prediction is correct, then $y_if(x_i)>0$ ; if the prediction is wrong, then $y_if(x_i)=<0$, it can be seen that the loss function value of the wrong prediction is larger than the loss function value of the correct prediction, where $e^{-y_if(x_i)}$ Also in this round, the weight $w_{mi}$ of each sample shows that the weight value of each misclassification is larger, we minimize the loss function $L(y,f(x))$, that is, we will give priority to reducing the prediction error, that is, we must find a minimum $\alpha_m$ in each round of iterations so that $L(y,f(x))$ is the smallest, which is:

<font color="#dd000">$argmin_{\alpha_m}L(y,f(x))=argmin_{\alpha_m}[\frac{1}{N}\sum_{i=1}^N e^{-y_if(x_i)}]=argmin_{\alpha_m}\sum_{i=1}^Ne^{-y_i[f_{m- 1}(x_i)+\alpha_mG_m(x_i)]}$</font>

##### 1.3.2 Why is the classifier coefficient $\alpha_m=\frac{1}{2}ln\frac{1-e_m}{e_m}$ obtained in each round of the above algorithm?

We know that the loss function is: $L(y,f(x))=\frac{1}{N}\sum_{i=1}^N e^{-y_if(x_i)}$

$ L(y,f(x))=\frac{1}{N}\sum_{i=1}^N e^{-y_if(x_i)}=\sum_{i=1}^Ne^{-y_if_{m-1}(x_i)}\cdot e^{-y_i\alpha_mG_m(x_i)}$, $ w_{mi}=e^{- y_if(x_i)}$,

And $$ if the classification is correct, then $y_iG_m(x_i)=1$, if the classification is wrong, then $y_iG_m(x_i)=-1$,

$ L(y,f(x))=\sum_{i=1}^Nw_{mi}\cdot e^{-y_i\alpha_mG_m(x_i)}=\sum_{y_i=G_m(x_i)}w_{mi}e^{-\alpha_m}+\sum_{y_i\neq G_m(x_i)}w_{mi}e^{\ alpha_m}$

$=e^{-\alpha_m}\big(\sum_{i=1}^Nw_{mi}-\sum_{y_i\neq G_m(x_i)}w_{mi}\big)+\sum_{y_i\neq G_m(x_i)}w_{mi}e^{\alpha_m}=e^{-\alpha_m}\sum_{i=1}^N w_{mi}+(e^{\alpha_m}-e^{-\alpha_m})\sum_{y_i\neq G_m(x_i)}w_{mi}$

The reason why we do this is to express the loss function with error points as much as possible. In order to minimize the loss function, we will derive the loss function with respect to $\alpha_m$:

$ \frac{\partial L}{\partial \alpha_m}=-e^{-\alpha_m}\sum_{i=1}^Nw_{mi}+(e^{\alpha_m}+e^{-\alpha_m})\sum_{y_i\neq G_m(x_i)}w_{mi}=0$

$ -\sum_{i=1}^Nw_{mi}+(e^{2\alpha_m}+1)\sum_{y_i\neq G_m(x_i)}w_{mi}=0$,

$ e^{2\alpha_m}=\frac{\sum_{i=1}^Nw_{mi}}{\sum_{y_i\neq G_m(x_i)}w_{mi}}-1=\frac{\sum_{i=1}^Nw_{mi}-\sum_{y_i\neq G_m(x_i)}w_{mi}}{\sum_ {y_i\neq G_m(x_i)}w_{mi}}=\frac{\sum_{y_i= G_m(x_i)}w_{mi}}{\sum_{y_i\neq G_m(x_i)}w_{mi}}=\frac{1-e_m}{e_m}$

<font color="#dd000">$ \alpha_m=\frac{1}{2}ln\frac{1-e_m}{e_m}$</font>

So far, we have deduced the process of $ \alpha_m=\frac{1}{2}ln\frac{1-e_m}{e_m}$, which is actually the coefficient obtained by minimizing the loss function. <font color="#dd000">This coefficient has two functions:</font>

<font color="#dd000">①. Minimize error loss</font>

<font color="#dd000">②. The smaller the loss of a weak classifier, the greater the weight</font>

##### 1.3.3 Why is the sample weight $w_{m+1,i}=\frac{w_{mi}\cdot e^{-\alpha_my_iG_m(x_i)}}{\sum_{i=1}^Nw_{mi}\cdot e^{-\alpha_my_iG_m(x_i)}}$ in the above algorithm, how is it obtained?

It can be seen from the above: $w_{m+1,i}=e^{-y_if_m(x_i)}=e^{-y_i(f_{m-1}(x_i)+\alpha_mG_m(x_i))}=e^{-y_if_{m-1}(x_i)}\cdot e^{-y_i\alpha_mG_m(x_i)}=w_{mi} e^{-y_i\alpha_mG_m(x_i)}$

That is to get $w_{m+1,i}=w_{mi}e^{-y_i\alpha_mG_m(x_i)}$, here we need a classification discussion to understand:

First of all, we know that because the classification error rate of a weak classifier is always a little better than random guessing, so we select a certain segmentation point as a weak classifier in one round, then we must choose a classification error rate less than $\frac{1}{2}$, that is, limit $e_m<\frac{1}{2}$, that is, the error rate is less than $50\%$

①If $y_i=G_m(x_i)arrow w_{m+1,i}=w_{mi}e^{-\alpha_m}$, the judgment is correct, then:

 At this time, from $e_m<\frac{1}{2}$, that is, the error rate is less than $50\%$, then according to $\alpha_m=\frac{1}{2}ln\frac{1-e_m}{e_m}$, it can be seen that $\alpha_m>0$, the weight of the classifier at this time is large, and more importantly, $w_{m+1,i}$ means that the weight of the next round of correct samples will become smaller;

②If $y_i\neq G_m(x_i)arrow w_{m+1,i}=w_{mi}e^{\alpha_m}$, the judgment is wrong:

 At this time, from $e_m<\frac{1}{2}$, that is, the error rate is less than $50\%$, then according to $\alpha_m=\frac{1}{2}ln\frac{1-e_m}{e_m}$, it can be known that $\alpha_m>0$, the weight of the classifier at this time is large, and more importantly, $w_{m+1,i}$ means that the weight of the next round of error samples will become larger;

<font color="#dd000">This is the magic of the $AdaBoost$ serial adaptive algorithm, which always increases the weight of misclassified samples in each round, so that the next round will pay more attention to those wrong samples. </font>

Then $\sum_{i=1}^Nw_{mi}\cdot e^{-\alpha_my_iG_m(x_i)}$ on the bottom surface is relatively simple, it is a normalization factor, in order to make $w_{m+1,i}$ become a probability, it is only for this function.

Since then, we know why the next round of weight $w_{m+1,i}$ is equal to $\frac{w_{mi}\cdot e^{-\alpha_my_iG_m(x_i)}}{\sum_{i=1}^Nw_{mi}\cdot e^{-\alpha_my_iG_m(x_i)}}$.

#### 1.4. The derivation of the upper bound of the final classifier training error of the $AdaBoost$ algorithm, and the connotation of the upper bound

Suppose we get a final classifier $G(x)=signf(x)$ after training, then the training error of this final classifier is: $\frac{1}{N}\sum_{i=1}^NI(G(x_i)\neq y_i)$, this training error has an upper bound, the upper bound is:

<font color="#dd000">$\frac{1}{N}\sum_{i=1}^NI(G(x_i)\neq y_i)\leq\frac{1}{N}\sum_ie^{-y_if(x_i)}=\prod_mZ_m$</font>, where $Z_m=\sum_{i=1}^Nw_{mi}e^ {-\alpha_my_iG_m(x_i)}$ is the normalization factor

Proof: $ \frac{1}{N}\sum_{i=1}^NI(G(x_i)\neq y_i)=\frac{1}{N}(\sum_{y_i=G(x_i)}0+\sum_{y_i\neq G(x_i)}1)$,

$ 0<e^{-y_if(x_i)}$ and if the classification is incorrect, namely $y_i\neq G(x_i)$, then $1\leq e^{-y_if(x_i)}$, $ \frac{1}{N}\sum_{i=1}^NI(G(x_i)\neq y_i)\leq \frac{1}{N}( \sum_{y_i=G(x_i)}e^{-y_if(x_i)}+\sum_{y_i\neq G(x_i)}e^{-y_if(x_i)})=\frac{1}{N}\sum_ie^{-y_if(x_i)}$

That is to prove that <font color="#dd000">$\frac{1}{N}\sum_{i=1}^NI(G(x_i)\neq y_i)\leq\frac{1}{N}\sum_ie^{-y_if(x_i)}$</font>

And $ \frac{1}{N}\sum_ie^{-y_if(x_i)}=\frac{1}{N}\sum_ie^{-y_i\sum_{m=1}^M\alpha_mG_m(x_i)}=\frac{1}{N}\sum_ie^{\sum_{m=1}^M-\alpha_my_iG _m(x_i)}=\frac{1}{N}\sum_i\bigg[e^{-\alpha_1y_iG_1(x_i)}\cdot e^{-\sum_{m=2}^M\alpha_my_iG_m(x_i)}\bigg]$, and $w_{1i}=\frac{1}{N}$, $ \frac {1}{N}\sum_ie^{-y_if(x_i)}=\sum_iw_{1i}e^{-\alpha_1y_iG_1(x_i)}\cdot e^{-\sum_{m=2}^M\alpha_my_iG_m(x_i)}$

And $ w_{m+1,i}=\frac{1}{Z_m}w_{m,i}e^{-\alpha_my_iG_m(x_i)}$, $ w_{2,i}=\frac{1}{Z_1}w_{1,i}e^{-\alpha_1y_iG_1(x_i)}$, $ Z_ 1w_{2,i}=w_{1,i}e^{-\alpha_1y_iG_1(x_i)}$

$ \frac{1}{N}\sum_ie^{-y_if(x_i)}=Z_1\sum_iw_{2,i}e^{-\sum_{m=2}^M\alpha_my_iG_m(x_i)}$, similarly, $\frac{1}{N}\sum_ie^{-y_if(x_i)}=Z_1Z_2\ sum_iw_{3,i}e^{-\sum_{m=3}^M\alpha_my_iG_m(x_i)}=...=Z_1Z_2\cdot...\cdot Z_m=\prod_m Z_m$, which is $\frac{1}{N}\sum_ie^{-y_if(x_i)}=\prod_m Z_m$

<font color="#dd000">The proof is: $\frac{1}{N}\sum_{i=1}^NI(G(x_i)\neq y_i)\leq\frac{1}{N}\sum_ie^{-y_if(x_i)}=\prod_m Z_m$</font>



For the binary classification problem, the above $\prod_m Z_m$ also has an upper bound, which is:

$\prod_m Z_m=\prod_m2\sqrt{e_m(1-e_m)}=\prod_m\sqrt{1-4\gamma_{m}^2}\leq e^{-2\sum_m\gamma_{m}^2}$

Prove: $ Z_m=\sum_{i=1}^Nw_{mi}e^{-\alpha_my_iG_m(x_i)}=\sum_{y_i=G_m(x_i)}w_{mi}e^{-\alpha_m}+\sum_{y_i\neq G_m(x_i)}w_{mi}e^{\alpha_m}=( \sum_iw_{mi}-e_m)e^{-\alpha_m}+e_me^{\alpha_m}$

$=(1-e_m)e^{-\alpha_m}+e_me^{\alpha_m}\geq 2\sqrt{(1-e_m)e_m}$

It can be seen from the basic inequality that the above equal sign is equal if and only when $(1-e_m)e^{-\alpha_m}=e_me^{\alpha_m}$, at this time it happens that $\alpha_m=\frac{1}{2}ln\frac{1-e_m}{e_m}$ is the classifier coefficient obtained before, indicating that the inequality can be equal on the basis of the above algorithm.

$ Z_m=2\sqrt{(1-e_m)e_m}$, that is to prove <font color="#dd000">$\prod_m Z_m=\prod_m2\sqrt{e_m(1-e_m)}$</font>

And $ 2\sqrt{(1-e_m)e_m}=2\sqrt{e_m-e_m^2}=\sqrt{4e_m-4e_m^2}=\sqrt{1-(4e_m^2-4e_m+1)}=\sqrt{1-(2e_m-1)^2}$, let $\gamma_m=\frac{1}{ 2}-e_m$,

Then $2\sqrt{(1-e_m)e_m}=\sqrt{1-4\gamma_m^2}$, that is to prove <font color="#dd000">$\prod_m Z_m=\prod_m2\sqrt{1-4\gamma_m^2}$</font>

Now you only need to prove $\prod_m\sqrt{1-4\gamma_m^2}\leq e^{-2\sum_m\gamma_m^2}$, that is, you only need to prove $\prod_m\sqrt{1-4\gamma_m^2}\leq \prod_me^{-2\gamma_m^2}$, that is, you only need to prove $\sqrt{1-4\gamma_m^2} \leq e^{-2\gamma_m^2}$,

Then it proves that $\sqrt{1-4x^2}\leq e^{-2x^2}$

First, Taylor expands $\sqrt{1-4x^2}$ at $x=0$, that is, $\sqrt{1-4x^2}=1-\frac{2}{1!}x-\frac{4}{2!}x^2-\frac{24}{3!}x^3+...$

Then $e^{-2x^2}$ is Taylor expanded at $x=0$, that is, $e^{-2x^2}=1-\frac{2}{1!}x+\frac{4}{2!}x^2-\frac{8}{3!}x^3+...$

It can be seen that $\sqrt{1-4x^2}$ only has negative items, while $e^{-2x^2}$ has positive and negative items, and the negative items are relatively small, $ \sqrt{1-4x^2}\leq e^{-2x^2}$

$ \prod_m\sqrt{1-4\gamma_m^2}\leq e^{-2\sum_m\gamma_m^2}$

To sum up, <font color="#dd000">$\prod_m Z_m=\prod_m2\sqrt{e_m(1-e_m)}=\prod_m\sqrt{1-4\gamma_{m}^2}\leq e^{-2\sum_m\gamma_{m}^2}$</font>

So what does this upper bound mean?

We take the training error $\frac{1}{N}\sum_{i=1}^NI(G(x_i)\neq y_i)$ of the final classifier an exponential upper bound $e^{-2\sum_m\gamma_{m}^2}$, which shows that the training error of the $AdaBoost$ algorithm decreases at an exponential rate.



### 2. Practical part

#### 2.1. The basic idea of face detection

The basic idea of $AdaBoost$ face detection is divided into three parts. First, the first step is to quickly calculate the feature rectangle of the face data in the training set by using the $Harr-like$ wavelet feature and the integral graph, and obtain a massive feature data set; the second step is to use the $AdaBoost$ algorithm to calculate the best of the same features of all samples, that is, to obtain the optimal feature; the third step is to use the $AdaBoost$ cascade classifier to detect any image and identify the face.

##### 2.1.1. Using $Harr-like$ wavelet features and integral graphs

First of all, it is easier to understand that each area of the face has light and shade changes. It is easier for us to conclude that the area of the eyes is always darker than the color of the cheeks below the eyes. The sum of pixels in these two areas is significantly different from that of non-human faces, and the color of the eyes is always darker than the middle of the eyes, as shown in the following figure:

<img src="C:\Users\56966\AppData\Roaming\Typora\typora-user-images\image-20221206180044707.png" alt="image-20221206180044707" style="zoom: 50%;" />

We use this rectangular frame to calculate the pixel difference between the white rectangle and the black rectangle, which is a feature value that can be compared with non-face data. Of course, these rectangular features are not only these two, such as the pixel values of the mouth and around the mouth. $8$ Rectangular features:

<img src="C:\Users\56966\AppData\Roaming\Typora\typora-user-images\image-20221206180628044.png" alt="image-20221206180628044" style="zoom: 50%;" />

Among them, $ A1 $ and $ A2 $ are edge features, $ B1 $, $ B2 $, $ B3 $, $ B4 $ are linear features, $ C1 $ is rotating features, $ d1 $ is the center of characteristics. Each feature rectangle is a picture from left to right, according to the arbitrary scale and any proportion, respectively Go to calculate the pixel difference between the white rectangular area and the black rectangular area **. Because it is any location, any scale, and any proportion to scan a picture. The rough scanning process is as follows:

![image-20221206185813945](C:\Users\56966\AppData\Roaming\Typora\typora-user-images\image-20221206185813945.png)

So even a picture of $24\times24$ pixels, a rectangular feature, will generate about $7000$ million pieces of data under the $5000$ sample set, so how to quickly extract these features is a problem.

Therefore, according to "$Rapid\ Object\ Detection\ Using\ a\ Boosted\ Cascade\ of\ Simple\ Features$", a method of integral graph is proposed. We first calculate the value of the pixel sum in the upper left corner of each pixel of a picture, as the value of the pixel point, which is the integral graph. Then the pixel sum of any area of the picture is composed of the linear combination of the pixel sum of $4$ partial area, as shown in the figure below:

<img src="C:\Users\56966\AppData\Roaming\Typora\typora-user-images\image-20221206182025979.png" alt="image-20221206182025979" style="zoom:50%;" />

For example, the sum of pixels in the $D$ area is $Sum(D)=4-2-3+1$. In this way, as long as we calculate the integral map in advance, it is constant time to calculate the eigenvalue of any rectangular feature. The following is the calculation logic of each feature rectangle:

![image-20221206183917564](C:\Users\56966\AppData\Roaming\Typora\typora-user-images\image-20221206183917564.png)

![image-20221206184053829](C:\Users\56966\AppData\Roaming\Typora\typora-user-images\image-20221206184053829.png)



##### 2.1.2. Use the $AdaBoost$ algorithm to calculate the best of the same features of all samples

After extracting all the feature points, it is a huge file. In fact, any feature rectangle value with any width and height at any position is a feature point. However, since the sample set has a certain amount of $N$, we need to find an optimal segmentation point for a certain feature position in all samples, which is to sort the feature values of $N$ samples at a certain feature position from small to large. , this segmentation point is the first step of using the $AdaBoost$ algorithm to find the segmentation point corresponding to the smallest error. If we find all the segmentation points, it is equivalent to combining $5000$ samples into one, and the amount of data will be greatly reduced.

##### 2.1.3. Use $AdaBoost$ cascade classifier to detect any picture and recognize faces

After finding the optimal segmentation point of all features, the amount of data is still huge. If you randomly give a picture at this time, you still need to set up a detection window, scan the picture from left to right, from top to bottom, and then zoom in after scanning. For this detection window, use $AdaBoost$ to find weak classifiers in each round of each scan. If all segmentation points are used, the efficiency of face detection will be extremely low, and $AdaBoost$ was the fastest face detection method before deep learning became popular, so it is proposed at this time a cascading approach.

It is a detection window at any position on any scale in any picture. Non-human faces are very common. It is easy for us to judge a non-human face, but it is difficult to judge a human face, so at this time we can select several features that are easiest to judge a human face, such as $2$ and $3$, as a very lightweight classifier.

Therefore, the idea of the cascade classifier is to disperse a large number of optimal features, and only calculate the best features first. If the face condition is satisfied, the next step is judged. The cascade classifier method can save $95\%$. The approximate calculation logic is as follows:

<img src="C:\Users\56966\AppData\Roaming\Typora\typora-user-images\image-20221206185047756.png" alt="image-20221206185047756" style="zoom:67%;" />

The features of the detected face that are actually used are roughly as follows:

<img src="C:\Users\56966\AppData\Roaming\Typora\typora-user-images\image-20221206185558280.png" alt="image-20221206185558280" style="zoom: 50%;" />
