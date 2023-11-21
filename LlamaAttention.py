class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):        # config是一个LlamaConfig对象，它包含了Llama模型的配置信息，包括隐藏层大小、注意力头数等
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size       # 默认为4096, 可以理解为embed_dim, 为最后一个维度。
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # 在config初始化的时候
        # if num_key_value_heads is None:
        #     num_key_value_heads = num_attention_heads
        
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 比如如果self.num_key_value_heads = 8，那么self.num_key_value_groups = 32//8 = 4
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
            
        # 用于查询张量的线性投影层，将隐藏状态投影到注意力头的维度，实际上self.head_dim * self.num_heads == self.hidden_size。
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        
        # 用于键张量的线性投影层，将隐藏状态投影到键值头的维度。
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        
        # 用于值张量的线性投影层，将隐藏状态投影到键值头的维度。
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        
        # 用于输出的线性投影层，将注意力头的输出投影回隐藏层的维度。
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # 初始化RoPE（Rotary Position Embedding）层，用于处理位置嵌入。
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    # 用于将张量重新整形为适用于多头注意力的形状:tirch.Size([(bsz, self.num_heads, seq_len, self.head_dim)])
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,   # tuple中分别为k和v
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:    
        bsz, q_len, _ = hidden_states.size()

        # 通过线性投影将隐藏状态投影到查询、键和值的维度
        if self.config.pretraining_tp > 1:      # 将config.pretraining_tp的值设置为1以外的其他值时，将启用更准确但速度较慢的线性层计算方式，以更好地匹配原始的逻辑回归结果
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 加上旋转编码(两种情况：是否考虑过去的k_v_state)
        kv_seq_len = key_states.shape[-2]       # 即为q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]   # 如果要考虑past_key_value的影响，则要加上past_key_value_length，对应着.shape[-2]
            
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # print((query_states - query_states_init)[0, 0, :18*5, :10])

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # use_cache是一个布尔类型的参数，用于指示模型是否应该返回最后的键/值注意力（key/values attentions）。需要注意的是，该参数只在部分模型中使用，并且只在config.is_decoder=True时才相关。
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # (bsz, head, seq_length, head_dim) matmul (bsz, head, head_dim, seq_length) -> (bsz, head, seq_length, seq_length)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
    
    

        # upcast attention to fp32
        # print("before_softmax:", attn_weights)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
        # print("after_softmax:", attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)      # (bsz, head, seq_length, head_dim)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()      # (bsz, seq_length, head, head_dim)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)     # (bsz, seq_length, embed_dim)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
            # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
            # attn_output.shape = torch>Size([bsz, seq_length, embed_dim])

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
