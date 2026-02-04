import os


def run(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            k = k[len('model.'):]
        # Emb
        k = k.replace('embed_tokens', 'tok_embeddings')        
        # Attention
        k = k.replace('self_attn.q_proj', 'attention.wq')
        k = k.replace('self_attn.k_proj', 'attention.wk')
        k = k.replace('self_attn.v_proj', 'attention.wv')
        k = k.replace('self_attn.o_proj', 'attention.wo')
        k = k.replace('self_attn.q_norm', 'attention.q_norm')
        k = k.replace('self_attn.k_norm', 'attention.k_norm')
        # FFN
        k = k.replace('mlp.gate_proj', 'feed_forward.w1')
        k = k.replace('mlp.up_proj', 'feed_forward.w3')
        k = k.replace('mlp.down_proj', 'feed_forward.w2')
        # Norm
        k = k.replace('input_layernorm', 'attention_norm')
        k = k.replace('post_attention_layernorm', 'ffn_norm')
        new_state_dict[k] = v
    if new_state_dict.get('output.weight', None) is None:
        new_state_dict['output.weight'] = new_state_dict['tok_embeddings.weight']
    return new_state_dict
