# VirusDB
Virus AI platform

## Project significance 项目意义
2003年的SARS重症急性呼吸综合征、2009年H1N1型禽流感、2012的MERS中东呼吸综合征，2017年的H7N9型禽流感，以及2019年的SARS-Cov-2新型冠状
病毒，都导致大量人员感染，也导致一定人员死亡。作为ICT工作者，希望能够通过ICT技术(信息与通信技术)+BT技术(生物技术)帮助医疗工作者早发现、
早治疗、早隔离；帮助普通人员早知道、早预防、早治疗。


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
- [xuziyao26](https://github.com/xuziyao26)
- [lingyeyouming](https://github.com/lingyeyouming)
- [AllenZqy](https://github.com/AllenZqy)
- [DenseAi](https://github.com/denseai) 

## Thanks 致谢


## Reference 参考 
#### 知识图谱
- [OpenKG 新冠百科图谱](http://www.openkg.cn/dataset/2019-ncov-baike)
- [OpenKG 新冠科研图谱](http://www.openkg.cn/dataset/2019-ncov-research)
- [OpenKG 新冠临床图谱](http://www.openkg.cn/dataset/2019-ncov-clinic)

#### 病毒、基因、药物
- [Virus-Host DB](https://www.genome.jp/virushostdb)

#### 深度学习
- [kpot/keras-transformer](https://github.com/kpot/keras-transformer)
- [cerebroai/reformers](https://github.com/cerebroai/reformers)

#### 参考文献
- [Qian Guo, Mo Li, Chunhui Wang, Peihong Wang, Zhencheng Fang, Jie tan, Shufang Wu, Yonghong Xiao, Huaiqiu Zhu. Host 
and infectivity prediction of Wuhan 2019 novel coronavirus using deep learning algorithm.](https://doi.org/10.1101/2020.01.21.914044)
- [Peng Zhou, Xing-Lou Yang, Xian-Guang Wang, et al. Discovery of a novel coronavirus associated with the recent 
pneumonia outbreak in humans and its potential bat origin.](https://doi.org/10.1101/2020.01.22.914952)
- [Randhawa, G., Hill, K. & Kari, L. ML-DSP: Machine Learning with Digital Signal Processing for ultrafast, accurate, 
and scalable genome classification at all taxonomic levels. BMC Genomics 20, 267 (2019).](https://doi.org/10.1186/s12864-019-5571-y)
- [Fang, Z., Tan, J., Wu, S., Li, M., Xu, C., Xie, Z., and Zhu, H. (2019). PPR-Meta: a tool for identifying phages and 
plasmids from metagenomic fragments using deep learning. GigaScience, 8(6), giz066.](https://doi.org/10.1093/gigascience/giz066)
- [Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya. Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)
- [David R. Kelley, Yakir A. Reshef. Sequential regulatory activity prediction across chromosomes with convolutional neural networks.
bioRxiv 161851;](https://doi.org/10.1101/161851)
- [Gurjit S. Randhawa, Maximillian P.M. Soltysiak, Hadi El Roz, Camila P.E. de Souza, Kathleen A. Hill, Lila Kari. 
Machine learning analysis of genomic signatures provides evidence of associations between Wuhan 2019-nCoV and bat betacoronaviruses. 
bioRxiv 2020.02.03.932350;](https://www.biorxiv.org/content/10.1101/2020.02.03.932350v2.full)