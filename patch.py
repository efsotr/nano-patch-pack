import torch
from warnings import warn

def wrap(fn):
    
    def wrapped_fn(input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, **kwargs):
        assert attention_mask is not None
        assert not kwargs.get("output_hidden_states", False)
        
        # prepare shape and attention mask
        B, L = attention_mask.shape
        attention_mask = attention_mask.bool()

        # prepare input
        if input_ids is None:
            input_ids = input_ids[attention_mask][None] # (1, T)
        else:
            inputs_embeds = inputs_embeds[attention_mask][None] # (1, T, D)

        # prepare position ids
        if position_ids is None:
            position_ids = torch.arange(0, L, device=attention_mask.device)[None]
        position_ids = position_ids.expand(B, L).clone()[attention_mask][None]

        indices = torch.nonzero(attention_mask.view(-1), as_tuple=True)[0]

        # prepare flash attention 2 kwargs for transformers models
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0), value=0)
        attention_mask = None
        kwargs["cu_seq_lens_q"] = cu_seqlens
        kwargs["cu_seq_lens_k"] = cu_seqlens
        kwargs["max_length_q"] = max_seqlen_in_batch
        kwargs["max_length_k"] = max_seqlen_in_batch

        # original forward
        output = fn(input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    position_ids=position_ids, 
                    inputs_embeds=inputs_embeds, 
                    **kwargs) 

        # restore original shape 
        last_hidden_state = output.last_hidden_state # (1, T, H)
        output.last_hidden_state = torch.zeros((B * L, last_hidden_state.size(-1)), 
                                               dtype=last_hidden_state.dtype,
                                               device=last_hidden_state.device)
        output.last_hidden_state[indices] = last_hidden_state[0] # (T, H)
        output.last_hidden_state = output.last_hidden_state.view(B, L, -1)

        return output

    return wrapped_fn

nonsac_enable = False

def enable_nonsac(enable_remove_add=True):
    if not nonsac_enable:
        import torch._functorch.config
        import torch._functorch.partitioners as partitioners

        nonsac_enable = True

        # reduce activation memory footprint 
        torch._functorch.config.activation_memory_budget = 0.99

        if enable_remove_add:
            # Remove 'add' from the default operation list to eliminate redundant re-computation 
            # due to wrong partitioning of nodes
            def remove_add(fn):
                def wrapped_fn():
                    optypes = fn()
                    optypes.recomputable_ops.remove(torch.ops.aten.add)
                    return optypes
                return wrapped_fn

            partitioners.get_default_op_list = remove_add(partitioners.get_default_op_list)

# enable torch.compile memory consumption optimization
enable_nonsac()

def patch(model):
    # disable unnecessary kv cache
    if model.config.use_cache:
        warn("model use_cache is not False")
    model.config.use_cache = False

    # enable pack
    if model.config._attn_implementation != "flash_attention_2":
        warn("model attn_implementation is not flash_attention_2")
    model.config._attn_implementation = "flash_attention_2"
    model.model.forward = wrap(model.model.forward)

    # enable torch.compile
    for layer in model.model.layers:
        layer.forward = torch.compile(layer.forward, dynamic=True)