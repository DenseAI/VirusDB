# VirusDB
Virus AI platform

## Project significance 项目意义
2003年的SARS重症急性呼吸综合征、2009年H1N1型禽流感、2012的MERS中东呼吸综合征，2017年的H7N9型禽流感，以及2019年的SARS-Cov-2新型冠状
病毒，都导致大量人员感染，也导致一定人员死亡。作为ICT工作者，希望能够通过ICT技术(信息与通信技术)+BT技术(生物技术)帮助医疗工作者早发现、
早治疗、早隔离；帮助普通人员早知道、早预防、早治疗。

Severe Acute Respiratory Syndrome of SARS in 2003, H1N1 Avian Influenza in 2009, MERS Middle East Respiratory Syndrome 
in 2012, H7N9 Avian Influenza in 2017, and SARS-Cov-2 New Coronavirus in 2019, all caused a large number of people 
Infection has also led to the death of some people.
As an ICT worker, I hope that I can help medical workers to find, treat, and isolate early through ICT technology 
(information and communication technology) + BT technology (biotechnology); help ordinary people to know, prevent, 
and treat early.


## Project ideas 项目思路
- 收集病毒百科、科研、临床、事件等信息，形成病毒知识图谱，方便快速查阅。
- 收集病毒基因、宿主信息，通过深度学习训练，提供新病毒分类预测、宿主预测等。
- 收集病毒基因、药物、治疗信息，通过深度学习训练，提供新病毒治疗方法评估、药物评估。
- 收集病毒事件，形成新型冠状病毒动力学模型，评估传播风险，进行病毒预警。

## Related technologies 相关技术
- 知识图谱
    - 命名实体识别、实体抽取、事件抽取、关系抽取；
    - 知识存储：Neo4j
    - 知识查询：Cypher
    - 知识可信度评估：深度学习
- 病毒分类、宿主预测
    - 卷积神经网络
    - 图神经网络
    - 注意力模型
- 病毒药物、治疗方案评估
    - 知识图谱
    - 图神经网络
    - 注意力模型
- 病毒动力学模型
    - SIR
    - SEIR

## 项目计划
- 知识图谱
    * 完成病毒百科图谱，包括知识抽取、图形化展示。 2020-02-14 ~ 2020-02-23
- 病毒分类、宿主预测
    * 完成病毒预测模型，包括模型训练、WebService服务。 2020-02-14 ~ 2020-02-23


## Collaborators 合作者
[Author 1]()
[Author 2]()
[Author 3]()
[Author 4]()
[Author 5]()
[DenseAi](https://github.com/denseai) 

## Reference 参考 
#### 知识图谱
- [OpenKG 新冠百科图谱](http://www.openkg.cn/dataset/2019-ncov-baike)
- [OpenKG 新冠科研图谱](http://www.openkg.cn/dataset/2019-ncov-research)
- [OpenKG 新冠临床图谱](http://www.openkg.cn/dataset/2019-ncov-clinic)

#### 病毒、基因、药物
- [Virus-Host DB](https://www.genome.jp/virushostdb)

#### 深度学习
- [kpot/keras-transformer](https://github.com/kpot/keras-transformer)

#### 参考文献
- [Qian Guo, Mo Li, Chunhui Wang, Peihong Wang, Zhencheng Fang, Jie tan, Shufang Wu, Yonghong Xiao, Huaiqiu Zhu. Host 
and infectivity prediction of Wuhan 2019 novel coronavirus using deep learning algorithm.](https://doi.org/10.1101/2020.01.21.914044)
- [Peng Zhou, Xing-Lou Yang, Xian-Guang Wang, et al. Discovery of a novel coronavirus associated with the recent 
pneumonia outbreak in humans and its potential bat origin.](https://doi.org/10.1101/2020.01.22.914952)
- [Randhawa, G., Hill, K. & Kari, L. ML-DSP: Machine Learning with Digital Signal Processing for ultrafast, accurate, 
and scalable genome classification at all taxonomic levels. BMC Genomics 20, 267 (2019).](https://doi.org/10.1186/s12864-019-5571-y)
- [Fang, Z., Tan, J., Wu, S., Li, M., Xu, C., Xie, Z., and Zhu, H. (2019). PPR-Meta: a tool for identifying phages and 
plasmids from metagenomic fragments using deep learning. GigaScience, 8(6), giz066.](https://doi.org/10.1093/gigascience/giz066)