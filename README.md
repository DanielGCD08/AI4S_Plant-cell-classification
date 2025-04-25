2025 AI4S TEEN Cup Challenge Biology——区分三类植物细胞 https://bohrium.dp.tech/competitions/1331932611?tab=introduce
一.题目概述
根据显微镜拍摄的植物细胞图像区分植物细胞类别是中学生物的重要学习方向之一。随着AI技术的发展，王老师希望使用AI来代替中学生识别植物细胞类别。为此，王老师使用显微镜拍摄了若干张叶片表皮细胞（Epidermis Cell）、保护组织细胞（Dermal Tissue Cell）和分生组织细胞（Meristematic Tissue Cell）的图像。王老师希望你编写AI程序，让计算机根据显微镜图像自动识别这三种细胞。

二. 数据集
本题目包含训练集和测试集。

（1）训练集数据存储在”train“文件夹中。包含两个部分：显微镜下植物细胞图片存储在文件夹"image_train"，所有图片均为.jpg格式，大小均为80*60的灰度图，每张图片均有唯一编号；训练集的标签存储在"label_train.csv"文件中，其中第一列为"file_id"，为自然数，表示存储图片的编号；第二列为"label"，取值为0,1,2，对应三类细胞的类别，但是王老师也忘记了到底哪类细胞对应的哪个label的取值。在比赛过程中可以直接看到和下载训练集的数据：训练集。

（2）测试集数据存储在“test”文件夹中。包含两个部分，显微镜下植物细胞图片存储在文件夹"image_test"，所有图片均为.jpg格式，大小均为80*60的灰度图，每张图片均有唯一编号；测试集的图片编号存储在"label_test_nolabel.csv"文件中，只包含一列，为"file_id"，为自然数，表示存储图片的编号。在比赛过程中不能直接看到测试集的数据，只能通过baseline提供的方法进行接入：baseline.ipynb

三. 任务
（1）王老师使用显微镜拍摄了若干张叶片表皮细胞（Epidermis Cell）、保护组织细胞（Dermal Tissue Cell）和分生组织细胞（Meristematic Tissue Cell）的图像片存储在一个"data"文件夹中，数据存储的格式见”数据集“部分的相关说明。现在你需要基于训练集中的数据训练一个图像识别的模型对细胞进行分类，并在测试集上进行验证。将测试集中的图片的对应的label列在"label_test_nolabel.csv"中补齐。相比于单纯的AI工程师，你拥有更多的生物知识，所以王老师希望你能够结合中学生物学过的知识，在补齐label的过程中，不要将label输出为0,1,2，而是输出相应的细胞名称。叶片表皮细胞对应的label取值为Epidermis Cell，保护组织细胞对应的label取值为Dermal Tissue Cell，分生组织细胞对应的label取值为Meristematic Tissue Cell。
（2）使用CPU的训练时间+测试时间不能超过10分钟，连接时间和排队时间不计入总时间。
四. 提交
请提交submission.ipynb文件，其中包含训练模型的全部过程。
在submission.ipynb中将测试集的预测结果保存在submission.csv中，包含两列，第一列为"file_id"，对应图片的编号；第二列为"label"，取值为细胞名字，即Epidermis Cell, Dermal Tissue Cell, Meristematic Tissue Cell中的一个（注意：每个单词开头都大写，单词与单词之间有空格，最后一个单词后没有空格）。
环境：pytorch == 2.4.1+cpu
      python == 3.8.8
