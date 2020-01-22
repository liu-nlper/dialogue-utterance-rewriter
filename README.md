# Dialogue Utterance Rewriter

ACL 2019论文复现，多轮对话重写：[Improving Multi-turn Dialogue Modelling with Utterance ReWriter](https://www.aclweb.org/anthology/P19-1003.pdf)

## 1. 写在前面

### 1.1 为什么要复现？

    - 作者开源的代码是基于LSTM的，论文中基于Transformer的代码并未公布；
    - 论文实验所使用数据与公开的数据不一致，所以给出新的指标以供参考。

由于和作者沟通没有得到回应，所以不知道是否重现了作者的结果，代码和结论仅供参考，有问题欢迎一起交流讨论~

### 1.2 关于代码

代码是基于Google官方的[Transformer](https://github.com/tensorflow/models/tree/master/official/transformer)实现的，主要修改点包括：

  - `Encoder`：相比与原始的Transformer，输入端多了一个`segment`，也就是论文中的`turn embedding`；
  - `Decoder`：Decoder端的输出结果为两次dec-enc attention结果拼接，再过两层全连接层得到；
  - `Output Distribution`：在`./transformer/model/transformer.py`里封装了`DistributeLayer`类；
  - `beam search`：由于`DistributeLayer`的输出结果已经是概率值，所以beam search在记录得分时，直接对概率值取log即可；
  - 对中文词表的处理等。

### 1.3 关于数据

数据是从论文作者发布的[corpus.txt](https://github.com/chin-gyou/dialogue-utterance-rewriter/blob/master/corpus.txt)中获取，共包含2w个实例，**按顺序**将前18k作为训练集，剩余的2k作为开发集。

数据目录`./data`，包括：

  - train.txt：训练集，共18000条；
  - dev.txt：开发集，共2000条；
  - BLEU_REF.txt：开发集目标语言标注结果，每个句子占一行，用于在训练过程中计算开发集上的BLEU Score。

其中，训练集和开发集格式一致，共四列，中间使用单个制表符分隔，如下：

    question1 answer1 question2 question2_rewrited
    能给我签名吗 出专辑再议 我现在就要 我现在就要签名
    iphonex好不好 iphone不好用 为什么不好用 iphonex为什么不好用
    西安天气 西安今天的天气是多云转小雨25度到35度东北风3级 明天有雨吗 西安明天有雨吗
    秦始皇活了多久 50岁我确定 为什么 为什么确定秦始皇活了50岁

### 1.4 需要的额外资源

**字表文件**

训练使用的字表文件，可以直接使用中文BERT的字表，下载地址[vocab.txt](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)，解压得到的vocab.txt放置在`./resource`下；

需要注意的是，vocab.txt需包含以下特殊字符：

    [PAD]
    [EOS]
    [UNK]
    [CLS]
    [SEP]

**预训练BERT（可选的）**

模型的Encoder端参数可使用预训练的BERT进行初始化，参考2.1节。

## 2. 使用

### 2.1 训练模型

参数、路径已经有默认的设置，可以直接跳到Step 3进行训练，若需要修改参数，可参考Step 1-2。

#### Step 1：修改模型参数

修改文件`./transformer/model/model_params.py`：

    max_length_source：源语言最大长度，可根据数据实际长度分布进行调整。
    max_length_target：目标语言最大长度，可根据数据实际长度分布进行调整。
    vocab_size：词表大小，需和`./resource/vocab.txt`大小保持一致。
    hidden_size：Model dimension in the hidden layers.
    num_hidden_layers：Number of layers in the encoder and decoder stacks.
    num_heads：Number of heads to use in multi-headed attention.
    filter_size：Inner layer dimension in the feedforward network.

配置文件将模型分为三类`tiny`、`base`和`big`，完成配置后，可在训练时指定类型名进行设置。

#### Step 2：修改训练脚本

修改文件`./scripts/train.sh`：

    PARAM_SET：模型类型，分为`tiny`、`base`和`big`三类；
    DATA_TRAIN：训练集路径；
    DATA_DEV：训练集路径；
    MODEL_DIR：模型保存路径；
    VOCAB_FILE：词/字表路径；
    BERT_CHECKPOINT：预训练BERT路径，注意BERT参数需要与编码端一致；若不需要，则设置为none；
    BLEU_SOURCE：待翻译数据，和`DATA_DEV`相同，无需设置；
    BLEU_REF：开发集目标语言标注结果，用于在训练过程中计算BLEU Score；
    TRAIN_EPOCHS：训练epoch数。

#### Step 3：训练

    $ cd scripts
    $ chmod a+x train.sh
    $ ./train.sh

### 2.2 测试

修改`./scripts/translate.sh`：

    PARAM_SET：与训练阶段保持一致；
    MODEL_DIR：模型路径，与训练阶段保持一致；
    VOCAB_FILE：词表文件，与训练阶段保持一致；
    FILE：待预测文件，格式同开发集；
    FILE_OUT：预测结果存放路径。

训练过程中，程序会在`MODEL_DIR`下保存模型，可以执行下述命令生成结果：

    $ chmod a+x translate.sh
    $ ./translate.sh

生成结果和待处理数据在同一目录，若待处理数据为`dev.txt`，则结果文件为`dev.out.txt`，每个句子占一行。

### 2.3 评价

评价脚本接受两个输入：

    $ python3 evaluate.py -g ../data/BLEU_REF.txt -t ../data/dev.out.txt

## 3. 实验

### 3.1 实验设置

实验数据如1.3节所述，Transformer参数设置为：

    hidden_size: 256
    num_hidden_layers：6
    num_heads：8
    filter_size：1024

**模型**：L-Ptr-λ和T-Ptr-λ结构均与论文中一致，T-Ptr-λ-BERT为使用预训练的BERT初始化编码端（L-6_H-256_A-8中文字BERT使用百科类数据预训练，若没有条件训练可忽略该项）。

### 3.2 实验结果

下表是在开发集上最好的一个模型周围取三个模型，得分取均值，供参考：

| 模型 | BLEU-1 | BLEU-2 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L | EM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L-Ptr-λ | - | - | - | - | - | - | - |
| T-Ptr-λ | 88.5 | 84.8 | 77.1 | 92.7 | 85.0 | 89.0 | 52.6 |
| T-Ptr-λ-BERT | 89.6 | 86.5 | 79.9 | 93.5 | 86.9 | 90.5 | 57.5 |

PS: 1.8w的训练集对于L-6_H-256_A-8规模的模型来说还是太小了，针对该模型设计一些预训练任务同时预训练E-D端，应该会有进一步的提升。

注：基于LSTM的结果后续再补充。

## 4. Requirements

    tensorflow-gpu >= 1.13.0
