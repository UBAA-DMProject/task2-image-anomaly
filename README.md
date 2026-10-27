# 任务二 图像异常检测

北航《数据挖掘导论》2026课程大作业的必做任务二。已完成问题定义、图像特征、检测模型、一次固定协议实验、独立核验及报告，覆盖2.0—2.3。任务是分别判断hazelnut和zipper图像的正常/异常，输出连续异常分数与二值标签。

代码仓库：[UBAA-DMProject/task2-image-anomaly](https://github.com/UBAA-DMProject/task2-image-anomaly)。

## 课程资料核对

已检查四个课程目录，依据见 [资料核对与方法复核](reports/materials_review.md)，逐文件清单和哈希见 [materials_audit.json](reports/materials_audit.json)。

| 目录 | 实际内容 | 本任务中的用途 |
| --- | --- | --- |
| Image_Anomaly_Detection | 两类图片、正常/异常训练文件夹、102条测试图标签 | **唯一实际使用的图片与评价标签来源** |
| Cluster | 六类各100张图及聚类评价标签 | 可参考图像读取和HOG表示，未混入其图片或标签 |
| thyroid | 六维数值特征、正常训练集及带标签测试集 | 可参考正常近邻评分与阈值结构，未使用其数值数据 |
| FinanceQA | 数据库、年报、问答协议和独立评测材料 | 可选Agent任务，本项目不实施 |

复核结论：现有实现与任务二数据、标签和目标匹配，可以保留。局部近邻在拉链上的低召回是实际局限，全部失败结果均被保留。

## 方法与标签使用

| 方法 | 表示与评分 | 拟合数据 | 阈值与参数 |
| --- | --- | --- | --- |
| 全局HOG近邻 | 8100维HOG；5个正常参考图的平均欧氏距离 | 每类160张正常图 | 40张正常验证图分数的95%分位数 |
| 局部HOG近邻 | 15×15个38维块；邻近位置的3近邻距离；最高12块取均值 | 每类160张正常图 | 同上，独立校准本方法阈值 |
| HOG线性SVM | 全局HOG＋训练期StandardScaler＋LinearSVC | 每类160正常＋40异常 | 验证AP选C，验证F1选阈值 |

两个近邻方法是正常样本学习，SVM是使用额外异常标签的有监督对照。任务二提供train/bad，使用这些训练标签有数据依据；不能把SVM称为无监督方法，也不能将训练信息不同的结果当作纯算法优劣证明。

图像转灰度并缩放至256×256，提取9方向HOG，单元16×16、块2×2、L2-Hys归一化。局部描述符在36维HOG后加入块像素均值与标准差。原图保持不变。

## 数据划分与测试边界

协议先写入 [experiment_protocol.json](experiment_protocol.json)。固定种子42，对每类的正常、异常训练图片分别按80%/20%划分：

| 类别 | 拟合正常/异常 | 验证正常/异常 | 测试正常/异常 |
| --- | --- | --- | --- |
| hazelnut | 160 / 40 | 40 / 10 | 40 / 15 |
| zipper | 160 / 40 | 40 / 10 | 32 / 15 |

全部模型、参数和阈值冻结保存后，先生成无标签测试预测，再加载测试JSON评价；事件顺序见results/main_20261027/events.json。未按测试效果更改方法、阈值或主模型，未合并验证集重训。数据未发现完全相同文件或解码像素，但不能据此证明物体实例独立。

## 实际结果

2026-10-27运行，正类为异常1，判定规则为score > threshold。AP使用Average Precision；另保存PR曲线梯形面积AUPRC，二者不混称。

| 类别 | 方法 | AUROC | AP | Precision | Recall | F1 | FP / FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hazelnut | 全局HOG近邻 | 0.8450 | 0.6795 | 1.0000 | 0.0667 | 0.1250 | 0 / 14 |
| hazelnut | 局部HOG近邻 | 0.9767 | 0.9475 | 0.8125 | 0.8667 | 0.8387 | 3 / 2 |
| hazelnut | HOG线性SVM | 0.5300 | 0.3183 | 0.3043 | 0.4667 | 0.3684 | 16 / 8 |
| zipper | 全局HOG近邻 | 0.7375 | 0.4866 | 0.0000 | 0.0000 | 0.0000 | 2 / 15 |
| zipper | 局部HOG近邻 | 0.8063 | 0.5737 | 0.4000 | 0.1333 | 0.2000 | 3 / 13 |
| zipper | HOG线性SVM | 0.9417 | 0.9440 | 0.7368 | 0.9333 | 0.8235 | 5 / 1 |

局部近邻是实验前指定的主要正常样本检测器，在榛子上较好，但拉链漏检13/15。SVM在拉链上较好、在榛子上较弱。没有根据测试结果拼接各类别最佳方法后作为新的独立测试结论。

局部近邻的宏平均AUROC/AP/F1为0.8915/0.7606/0.5194，仅为两个类别的算术平均。热力图只有块级定性意义，课程没有提供像素掩码，不报告像素级准确率。

## 环境与完整运行

实际环境：Windows、Python 3.14.5、CPU，计算线程4。NumPy 2.5.4、SciPy 1.18.1、scikit-learn 1.9.1、scikit-image 0.26.0、Pillow 13.0.0、Matplotlib 3.11.2。完整版本见 [requirements-lock.txt](requirements-lock.txt)。

在本仓库根目录使用Windows PowerShell：

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest -v test_experiment
.\.venv\Scripts\python.exe run_experiment.py --output-dir results/reproduction
.\.venv\Scripts\python.exe verify_experiment.py --result-dir results/reproduction
```

本次实际运行目录为results/main_20261027。结果目录已存在时拒绝覆盖，复现应使用新目录。默认数据根目录按脚本位置解析为：

```text
../../DM_2026_Dataset/Image_Anomaly_Detection/Image_Anomaly_Detection
```

也可显式指定：

```powershell
.\.venv\Scripts\python.exe run_experiment.py `
  --data-root 'C:\path\to\Image_Anomaly_Detection' `
  --output-dir results/custom_run
```

数据根目录应包含hazelnut/、zipper/和image_anomaly_labels.json。训练标签来自train/good、train/bad；测试标签按类别＋相对路径匹配，不能只按重复文件名匹配。程序不向原始数据目录写结果。

## 加载模型预测

以下命令已实际运行验证。新图像评分只需要模型和图片，不需要标签或重新拟合：

```powershell
.\.venv\Scripts\python.exe predict.py `
  --model results/main_20261027/models/hazelnut_local_hog_knn.joblib `
  --input-path ../../DM_2026_Dataset/Image_Anomaly_Detection/Image_Anomaly_Detection/hazelnut/test/001.png `
  --output-csv results/new_prediction.csv
```

input-path也支持图片目录；调用者须确保图片属于模型对应的物体类别。已有CSV拒绝覆盖。仅加载本项目可信来源的joblib模型。

## 产物与报告

| 路径 | 内容 |
| --- | --- |
| run_experiment.py | 数据检查、特征、拟合、校准、冻结、预测和评价 |
| plot_results.py | ROC/PR、混淆矩阵、分数分布和案例图 |
| predict.py | 保存模型的独立推理 |
| test_experiment.py | 10项针对性单元测试 |
| verify_experiment.py | 独立指标重算、模型重载、全部测试图评分和哈希核验 |
| results/main_20261027/ | 6个模型、清单、划分、参数、预测和图表 |
| reports/task2_image_anomaly_report.pdf | 6页实验报告，覆盖2.0—2.3 |
| reports/task2_report.md | 可编辑报告正文 |
| reports/main_20261027/ | 小组共享的指标、预测、环境和图表快照 |
| reports/materials_review.md | 四个课程资料目录用途和方法适用性复核 |

报告已逐页渲染检查。报告采用学术报告结构，供小组合并；尚未将原始.doc模板逐项套版，作者及小组信息由提交时统一补充。build_report.py用于本次固定实验的排版和快照整理，修改实验设计后须同步复核叙述及案例。

原始数据、环境、缓存和完整results不纳入Git；代码、报告与较小证据快照在本仓库统一管理，报告及证据保留在reports。

## 验证与局限

- 10项单元测试、依赖检查均通过。
- 独立重载6个模型，重新解码102张测试图，核对306条预测；分数最大差异0。
- 按正负样本两两比较重算AUROC，按分组排序重算AP，按整数计数核对分类指标。
- 602张原图哈希一致，冻结顺序、正常验证分位数、SVM缩放器拟合范围均通过核验。
- 单次划分、每类仅15张测试异常、灰度化和下采样、正常尾部分位数及小样本选参均限制结论。
- 后续若利用本次测试发现继续优化，应明确探索性质，或另找未参与选择的数据评价。
