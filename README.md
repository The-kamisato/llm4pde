# llm4pde

complete_model是四个网络的组合, 其中self.llama会映射到LlamaForCausalLM这个model, 这个model里面的sel.model是LlamaModel, 做的修改是在LlamaModel里面进行的。

modelling_llama里面带有注释#“pdepdepdepdepde”的为更改代码的标记

几个实验：time_embed = True, pos_embed = True可以控制是否建立可学习的(time_embed)和(pos_embed):

以可学习的(time_embed)为例：

在LlamaModel.py文件里面，14行部分是初始化(time_embedding)


