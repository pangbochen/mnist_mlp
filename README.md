# mnist_mlp
mnist mlp in pytorch and tensorflow

utils.py 

	实现一些辅助函数

config.py

	定义模型的参数
	我将MLP模型的参数分为了两类：训练参数与网络参数
		训练参数：
			使用python的argparser
			Namespace(batch_size=100, cuda=False, epochs=10, log_interval=300, lr=0.01, momentum=0.5, seed=6, test_batch_size=100, test_interval=5)
			包括了batch_size， 迭代次数，log参数， 随机种子等
		神经网络参数：
			使用python的class类
			{'n_hiddens': [512, 256, 128, 64], 'type_act': 'relu', 'rate_dropout': 0.2}
			包括网络的隐单元结构，激活函数种类，dropout等参数

model.py

	定义模型的结构
	模型的input_size为28*28=784
	每一个隐藏层分别由线性链接，激活函数，dropout三部分组成
	输出为线性层output_size为10
	使用了nn.Linear和nn.Sequencial函数

main.py

	实验的主代码，分为下面步骤：
		get args
		load dataset：使用pytorch提供的mnist数据集API
		get MLP model
		set loss function
		train model：训练的迭代次数由args.epochs设置
		test model：每过args.test_interval的训练迭代次数会进行测试，并保留效果最好的模型

autoencoder.py
  
  使用autoencoder来预训练模型，提升模型速率
  
MNIST

  98.44%
  
TF
  使用tensorflow和tensorboard实现模型
