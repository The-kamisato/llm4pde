@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.n_token_per_time = config.n_token_per_time          # pdepdepdepdepdepdepdepdepde
        if config.time_embed:
            # self.time_embedding = nn.Parameter(torch.randn(config.max_position_embeddings, config.hidden_size))   # pdepdepdepdepdepde
            self.time_embedding = nn.Parameter(torch.empty(50, config.hidden_size), requires_grad=False)     # pdepdepdepdepdepde
            nn.init.kaiming_normal_(self.time_embedding)
        else:
            self.time_embedding = None
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    # 注意力掩码是在进行自然语言处理任务时常用的一种技术，用于在处理序列数据时限制模型只能关注前面的标记，以保持语法和语义的正确性
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:                 # input_shape[-1]即为seq_len
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    # 以下是在llama for pde中使用的mask生成函数：# pdepdepdepdepdepdepdepdepde
    def _prepare_decoder_attention_mask_for_pde(self, attention_mask, input_shape, inputs_embeds, past_key_values_length, n_token_per_time):
        # create causal mask(根据输入的inputs_embeds的shape:input_shape, 这里只需要前两个维度大小bsz和seq_length)
        # [bsz, seq_len, embed_dim] -> [bsz, 1, T*n_token_per_time, T*n_token_per_time]
        combined_attention_mask = None
        combined_attention_mask = _make_causal_mask_for_pde(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
                n_token_per_time = n_token_per_time
            )

        if attention_mask is not None:      # 检查是否提供了额外的注意力掩码
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    def _prepare_earth_specific_pos(self, t_length, device):
        length_theta = math.floor(math.sqrt(self.n_token_per_time / 2))         # 3
        length_phi = self.n_token_per_time // length_theta         # 6
        
        theta_grid = math.pi * torch.arange(start = -1 / 2 + 1 / (2 * length_theta), end = (1.0001 / 2) - 1 / (2 * length_theta), step = 1 / length_theta)
        phi_grid = math.pi * torch.arange(start = -1 + 1 / length_phi, end = 1.0001 - 1 / length_phi, step = 2 / length_phi)
        
        phi_0 = phi_grid[0]
        theta_0 = theta_grid[0]
        
        x_0, y_0, z_0 = torch.cos(phi_0) * torch.cos(theta_0), torch.sin(phi_0) * torch.cos(theta_0), torch.sin(theta_0)
        x, y, z = torch.cos(phi_grid).unsqueeze(0) * torch.cos(theta_grid).unsqueeze(1), torch.sin(phi_grid).unsqueeze(0) * torch.cos(theta_grid).unsqueeze(1), torch.sin(theta_grid).unsqueeze(1)
        distance = torch.sqrt((x - x_0)**2 + (y - y_0)**2 + (z - z_0)**2).to(device)
        distance = distance.reshape(self.n_token_per_time).repeat_interleave(t_length)
        return distance
        
    # inputs_embeds允许你直接传入一个嵌入表示（embedded representation），而不是传入input_ids。如果你希望对如何将input_ids索引转换为相关向量具有更多的控制权，那么使用inputs_embeds是很有用的。
    # 这个参数的形状为(batch_size, sequence_length, hidden_size)，是一个torch.FloatTensor类型的张量。
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,             # 输入序列的标记（token）ID。它是模型输入的主要部分，用于表示输入文本的token化形式
        attention_mask: Optional[torch.Tensor] = None,  # 用于指示哪些标记需要被注意力机制忽略的二进制掩码。它的形状与 input_ids 相同，其中 1 表示要注意的标记，0 表示要忽略的标记。
        position_ids: Optional[torch.LongTensor] = None,            # 用于指示每个标记在序列中的位置的ID。它的形状与 input_ids 相同
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 可以是一个空列表（None），或者是一个包含多个torch.FloatTensor元素的列表
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        # if position_ids is None:
        #     device = input_ids.device if input_ids is not None else inputs_embeds.device
        #     position_ids = torch.arange(
        #         past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        #     )
        #     position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        # else:
        #     position_ids = position_ids.view(-1, seq_length).long()
        
        # pdepdepdepdepdepdepdepdepde
        

        assert(seq_length % self.n_token_per_time == 0)
        t_length = seq_length // self.n_token_per_time
        
        assert(past_key_values_length % self.n_token_per_time == 0)
        t_past_length = past_key_values_length // self.n_token_per_time
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            
            position_ids_time = torch.arange(
                t_past_length, t_length + t_past_length, dtype=torch.long, device=device
            ).repeat_interleave(self.n_token_per_time, dim=0)
            
            # position_ids_earth_specific = self._prepare_earth_specific_pos(t_length, device = device)
            
            # position_ids = position_ids_time + 0.1 * position_ids_earth_specific
            
            position_ids = position_ids_time
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        # print("position_ids.shape:", position_ids.shape)        # (1, seq_length)
        # print("position_ids:", position_ids)
        # pdepdepdepdepdepdepdepdepde
        
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # pdepdepdepdepdepde
        # 使用 torch.nonzero() 函数获取 NaN 元素的索引
        # nan_mask = torch.isnan(self.time_embedding)
        # nan_indices = torch.nonzero(nan_mask)

        # print(nan_indices)
        
        if self.config.time_embed:
            time_embedding = self.time_embedding[None, :t_length, :].repeat(batch_size, self.n_token_per_time, 1)
            inputs_embeds = inputs_embeds + time_embedding 
        # pdepdepdepdepdepde
        
        
        
        
        # embed positions
        # if attention_mask is None:
        #     attention_mask = torch.ones(
        #         (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        #     )
        # attention_mask = self._prepare_decoder_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        # )
        
        # pdepdepdepdepdepdepdepdepde
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask_for_pde(
            attention_mask, inputs_embeds.size(), inputs_embeds, past_key_values_length, self.n_token_per_time
        )
        # print("attention_mask,shape", attention_mask.shape)
        # print("attention_mask", attention_mask)
        # pdepdepdepdepdepdepdepdepde
        

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                
                # 创建一个自定义的前向传播函数
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)    # (all_hidden_states, hidden_states)   

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,        # 最后一层的hidden_states
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
