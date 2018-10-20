# Training Logs

老师上次说是用5个100%训练集训练的网络作为Teacher Net来对Student Net做蒸馏。

## 网络结构：

![ideas](ideas_revised.png)

## sample_rate=1.0 的实验结果：

#### **1. 充分蒸馏**

* **Stage1:**对5个Teacher Nets 充分蒸馏(**480个epoch**) ，达到converge，如下图**深蓝色线**：

![loss_s1](s1_loss_r1.png)

   * **Stage2:**对Student Net进行常规训练：

      * **Case1:** 采用与平常训练相同的Learning Rate(深蓝和橙色曲线是正常训练的曲线作为对比)。

        ![loss_v1](s2_v1.png) 

        **结果:** 蒸馏后的top1_accuracy略低于正常训练（71.88<72.66)

      * **Case2:**取平常寻训练的1/10的Learning Rate进行训练

        ![s2_v2](s2_v2.png)

        ****

        **结果:** Student Net 在第20epoch时就已经达到正常训练的top1_accuracy. 

        Student Net的最好训练结果也高于正常训练 （72.95>72.66）

#### 2. 非充分蒸馏

* **Stage1:**对5个Teacher Nets 充分蒸馏(**100个epoch**) ，达到converge，如下图**浅蓝色线**：

  ![](s1_loss_r1.png)

* **Stage2:**对Student Net进行正常训练

  * **Case1:**  与正常训练相同的learning rate:

    ![](s2_v3.png)

    **结果:** top1_accuracy 低于正常训练 （70.89<72.66）

    此后又分别进行了增加train epoch和降低learning rate 到原来1/2的训练，结果与此大致相同。

  * **Case2：**1/10正常训练的learning rate.

    ![](s2_v4.png)

    **结果：**top1_accuracy 略低于正常训练（71.76<72.66）