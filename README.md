
# LLMProject for ZaowuyunAi written test
首先感谢各位老师给我本次笔试项目的创作机会。通过飞书文档和造物云官网的产品介绍，我对造物云的产品已经有了基本的认识。造物云是国内领先的 AI 设计平台，基于 AIGC技术为用户营造极致的 AI 造物新体验。   
  
本次测试我的项目选题为 基于Qwen2.5-0.5B预训练模型进行微调、优化和推理。
  
我的开发环境为Windows11，32GB内存，CPU为Intel Xeon E5-2666v3，GPU为Nvidia GTX 1660Super 6G。使用Python3.10、PyTorch 2.5.1，Cuda版本为11.8 。

## File structure

```
tree LLMProject
|-- dpo_dataset        # dpo数据集
|   |-- sharegpt.jsonl
|-- dpo_qwen           # dpo训练后的模型
|-- logs               # 训练日志
|-- Qwen2.5-0.5B       # Qwen2.5-0.5B预训练模型
|-- dpo.py             # dpo训练的代码
|-- inference.py       # 模型推理代码
|-- requirements.txt   # 项目依赖库
|-- run_dpo.bat        # dpo运行脚本
|-- sft_dataset.json   # sft数据集
|-- README.md
```




## Project step

- 从ModelScope下载Qwen2.5-0.5B预训练模型
```shell
git clone https://www.modelscope.cn/Qwen/Qwen2.5-0.5B.git
```
- 下载sft数据集和dpo数据集。（SFT数据集为自制，来源于本人开源代码库。DPO数据集来源于HuggingFace开源多轮对话数据集ShareGPT_Vicuna_unfiltered）
```shell
git clone https://www.wisemodel.cn/SMU_AI_Lab/Medical_SFT_Dataset.git
git clone https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
```
- 通过sft.py（直接在IDE运行代码）,使用sft数据集对Qwen2.5-0.5B预训练模型进行微调。以医学对话数据集为例，进行微调，注入医疗领域知识。      
- 通过dpo.py,使用dpo数据集对Qwen2.5-0.5B预训练模型进行直接偏好优化。使模型具有更强大的多轮对话能力。
```shell
run_dpo.bat
```
- 通过inference.py（直接在IDE运行代码）,使用优化后的模型进行推理。  
  
⭐Note：sft.py代码为本人自行编写，在编写过程中，我使用了字节豆包大模型和ChatGPT-4o帮助我修改一些error和bug。时间原因，dpo.py代码是Github上的开源代码。在代码编写中，使用了字节旗下代码补全工具MarsCode.
由于本人电脑配置较低，并且时间较为紧张，训练过程无法完成，所以很抱歉，本项目只能只给出框架代码，希望老师理解。

- 最后将整个项目文件通过git上传到个人Github仓库。为方便老师的访问，同步上传到了个人Gitee仓库。
  
⭐Note：抱歉！预训练模型文件过大，我的lfs空间不够，所以没有上传预训练模型文件.

## The End
再次感谢各位老师，让我有机会完成本次笔试项目。无论结果如何，这对我都是一次宝贵的锻经历。  

### Any questions  
张津瑞  18686560986  
ruiruiz2308@163.com
