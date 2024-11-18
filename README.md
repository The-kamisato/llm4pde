# llm4pde

## complete_model是四个网络的组合, 其中self.llama会映射到LlamaForCausalLM这个model, 这个model里面的sel.model是LlamaModel, 做的修改是在LlamaModel里面进行的。

modelling_llama里面带有注释#“pdepdepdepdepde”的为更改代码的标记（主要是在LlamaModel,LlamaAttention中修改）

## 几个实验：time_embed = True, pos_embed = True可以控制是否建立可学习的time_embedding和pos_embedding。如果time_embed = False, 则默认使用[1,1,1,...,2,2,2...]RoPE之后的结果。

注：pos_embedding我在LlamaAttention基础上改了一个新的LlamaEarthSpecificAttention, 不过在pos_embed=False的情况下LlamaEarthSpecificAttention不涉及forward，因此先解决更重要的bug.

1. 可学习的(time_embed)为例：

在LlamaModel.py文件里面，19-24行部分是初始化time_embedding， 193-196行是加上time_embedding

2. 在time_embed = False的基础上，删去默认的RoPE：

在LlamaAttention.py里面，删去104-106行。

3. 删去加在atten_matrix上面的mask：

   删去LlamaAttention.py里面的136 行。


We are grateful to the author for providing us with numerous suggestions regarding theory and analysis. We will supplement the formulas and conduct further in-depth analysis as well as add relevant experiments.

**W1, W2:** The paper lacks rigorous theoretical analysis of why their proposed MatryoshkaKV method works better than PCA-based approaches. 

We have further formalized our formulae. 
However, the mathematics involved in designing the inter-layer and intra-layer structures of LLM is highly intricate due to confounding effects. 
We regret that we are unable to provide rigorous mathematical proofs within a short timeframe.
To enable authors to comprehend our motivation and gain insight, we conduct an in-depth analysis of a single head within a single layer in Appendix A. We find that to achieve the minimum error, it is essential to jointly optimize the projection matrices for both K and V. 
For example, if the principal components of K and V are just lie in each other's orthogonal complement, the approximation error would be extremely large. 
This shows that methods like PCA, which do not consider the interaction between K and V, are suboptimal.
Unfortunately, due to the nonlinear relationship (softmax function) in the calculation between K and V within the attention mechanism, certain mathematical methods such as GSVD (Generalized Singular Value Decomposition) are inapplicable for jointly optimizing the projection matrices of both K and V.

Morever, the optimal solution also varies with the input data distribution theoretically. 
Given the difficulty in modeling the distribution of all corpora globally, we believe that using a data-driven approach for optimization is a reasonable way to minimize the error of the model after KV cache compression on most tasks. 
Specifically, we make these orthogonal matrices trainable to obtain optimal results. 
By directly fine-tuning the projection matrices to maximize the data likelihood, we can better adapt to different data distributions and task requirements, thereby enhancing the overall performance of the model.

**Q1:** Why does the Matryoshka training strategy work better than static compression ratios? What's the theoretical justification?

The static strategy has a significant drawback in that it suffers from poor generalization. 
Low-rank projections fail to adapt and perform effectively at compression ratios that are not encountered during the training phase. 
In contrast, our proposed strategy, which **employs randomly sampled compression ratios on each attention head during training process**, offers several notable advantages:
- The randomly sampled compression ratios during training endow the low-rank projections with the ability to generalize across all possible compression ratios. This enables users to arbitrarily choose KV cache compression ratios for tasks of different difficulties. We refer to this property as “hierarchical structures in orthogonal matrices” in the paper.
- Setting various compression ratios across difernt attention heads effectively decouples the working ratios of different heads. This means that during the testing phase, we have the flexibility to set different ratios for different heads, which aligns with the observed phenomenon of isotropy in different heads as reported in some related works we've discussed in Section 5.4.






