from scipy.stats import zscore
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import inspect
from functools import reduce
import types
from tqdm import tqdm

import util_model
import util
from compute_graph_patcher import *

class NeuronActivationExtractor:
    def __init__(self, model, tokenizer, model_config):
        """
        Initialize the extractor with the given model, tokenizer, file root, model name,
        and safe neuron threshold. If target_layers is not provided, it will auto-detect 
        the number of MLP layers via util.count_mlp_module.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        print("Patching model...")
        if model_config.model_name == "deepseek-moe-16b-chat":
            for layer in self.model.model.layers:
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                    layer.mlp.gate.forward = types.MethodType(deepseek_moe_gate_forward, layer.mlp.gate)
        elif model_config.model_name == "Qwen1.5-MoE-A2.7B-Chat":
            for layer in self.model.model.layers:
                if hasattr(layer, 'mlp'):
                    layer.mlp.forward = types.MethodType(qwen1_5_moe_forward, layer.mlp)
        elif model_config.model_name == "gpt-oss-20b":
            for layer in self.model.model.layers:
                if hasattr(layer, 'mlp'):
                    layer.mlp.forward = types.MethodType(GptOssMLP_forward, layer.mlp)
                    if hasattr(layer.mlp, 'router'):
                        layer.mlp.router.forward = types.MethodType(GptOssTopKRouter_forward, layer.mlp.router)
                    # Replace experts
                    if hasattr(layer.mlp, 'experts'):
                        old_experts = layer.mlp.experts

                        # Create new experts with per-expert gate-up layers
                        new_experts = GptOssExperts(self.model.config)
                        new_experts = new_experts.to(old_experts.down_proj.device)
                        new_experts = new_experts.to(old_experts.down_proj.dtype)

                        # Copy weights for down projection
                        new_experts.down_proj.data.copy_(old_experts.down_proj.data)
                        new_experts.down_proj_bias.data.copy_(old_experts.down_proj_bias.data)

                        # Copy weights for each expert's gate-up layer
                        for i in range(new_experts.num_experts):
                            new_experts.gate_up_layers[i].gate_up_proj.data.copy_(
                                old_experts.gate_up_proj.data[i]
                            )
                            new_experts.gate_up_layers[i].gate_up_proj_bias.data.copy_(
                                old_experts.gate_up_proj_bias.data[i]
                            )

                        # Replace the experts module
                        layer.mlp.experts = new_experts
        print("Patching finished")
        # print(model)
        self.model.eval()

    def get_module_by_name(self, model, module_name):
        return reduce(getattr, module_name.split('.'), model)
    
    def count_mlp_module(self, model, model_name):
        mlp_count = 0
        for name, module in model.named_modules():
            if any(keyword in name.lower() for keyword in ['gate', 'up']):
                mlp_count += 1
        # These two model has gate and up fused into one single layer
        if model_name == "phi-4" or model_name == "Phi-4-mini-instruct":
            return mlp_count
        else:
            return int(mlp_count/2)

    def get_safety_neurons(self, weights, safe_neuron_threshold=2):
        z_scores = zscore(weights)
        candidates = np.where((np.abs(z_scores) > safe_neuron_threshold) & (weights > 0))[0]
        return np.array(candidates)

    def hook_patching(self, layer_name, candidate_neurons, patching_vector, alpha=1):
        def prune_hook(module, input, output):
            if len(patching_vector) != len(candidate_neurons):
                raise ValueError(f"Length of patching_vector ({len(patching_vector)}) must match length of candidate_neurons ({len(candidate_neurons)})")
            
            if self.model_config.model_name == "deepseek-moe-16b-chat" and layer_name.endswith(self.model_config.name_gate):
                pruned_output = output[2]
            elif self.model_config.model_name=="Qwen1.5-MoE-A2.7B-Chat" and layer_name.endswith(".mlp"): 
                pruned_output = output[0]
            else:
                pruned_output = output   

            if len(candidate_neurons) > 0:
                if self.model_config.model_name == "gpt-oss-20b" and self.model_config.name_expert_layers[0] in layer_name:
                    gate, up = pruned_output
                    gate[..., candidate_neurons['gate']] = torch.tensor(patching_vector['gate'], dtype=torch.bfloat16, device=gate.device)
                    up[..., candidate_neurons['up']] = torch.tensor(patching_vector['up'], dtype=torch.bfloat16, device=gate.device)
                else:
                    pruned_output[..., candidate_neurons] = torch.tensor(patching_vector, dtype=torch.bfloat16, device=pruned_output.device)


            if self.model_config.model_name == "deepseek-moe-16b-chat" and layer_name.endswith(self.model_config.name_gate):
                scores = pruned_output.softmax(dim=-1)
                topk_weight, topk_indices = scores.topk(self.model_config.topk, dim=-1, sorted=False)  # [batch_size * seq_len, topk]
                return topk_indices, topk_weight, None
            elif self.model_config.model_name=="Qwen1.5-MoE-A2.7B-Chat" and layer_name.endswith(".mlp"):
                # Replace the second element with pruned_output, keep others unchanged
                output = list(output)
                output[0] = pruned_output
                return tuple(output)
            elif self.model_config.model_name == "gpt-oss-20b" and self.model_config.name_expert_layers[0] in layer_name:
                return gate, up
            else:
                return pruned_output
        return prune_hook

    def hook_prune(self, layer_name, candidate_neurons, prune_value=0):
        def hook(module, input, output):
            if self.model_config.model_name == "deepseek-moe-16b-chat" and layer_name.endswith(self.model_config.name_gate):
                out = output[2]
            else:
                out = output   
            
            if len(candidate_neurons) > 0:
                if self.model_config.model_name == "gpt-oss-20b" and self.model_config.name_expert_layers[0] in layer_name:
                    gate, up = out
                    gate[..., candidate_neurons['gate']] = prune_value
                    up[..., candidate_neurons['up']] = prune_value
                else:
                    out[..., candidate_neurons] = prune_value

            if self.model_config.model_name == "deepseek-moe-16b-chat" and layer_name.endswith(self.model_config.name_gate):
                scores = out.softmax(dim=-1)
                topk_weight, topk_indices = scores.topk(self.model_config.topk, dim=-1, sorted=False)  # [batch_size * seq_len, topk]
                return topk_indices, topk_weight, None
            elif self.model_config.model_name == "gpt-oss-20b" and self.model_config.name_expert_layers[0] in layer_name:
                return gate, up
            else:
                return out
        return hook
    
    def hook_get_act(self, layer_name, dim_reduction_method, neuron_indices=None):
        """
        Returns a hook function that extracts activations from the given neuron indices.
        """
        def hook(module, input, output):
            #for deepseek-moe-16b-chat, the mlp.gate output is: (selected_expert_index [6,], gate_output_logit [64,], None)
            if self.model_config.model_name=="deepseek-moe-16b-chat" and layer_name.endswith(self.model_config.name_gate): 
                output = output[2]
            elif self.model_config.model_name=="Qwen1.5-MoE-A2.7B-Chat" and layer_name.endswith(".mlp"): 
                output = output[0]
            
            if output.dim() == 2:
                output = output.view(self.current_batch_size, self.current_seq_len, output.shape[-1])
            
            max_acts = []
            for b in range(self.current_batch_size):
                # Get expert indices for the current batch item: (batch_size, seq_len, hidden_dim)
                if self.shared_weight is not None:
                    out = output[b] * self.shared_weight[b]
                else:
                    out = output[b]
                # Filter out expert indices corresponding to padding tokens (where mask is 0)
                real_out = out[self.attention_mask[b].to(out.device) == 1] 
                if dim_reduction_method == 'max':
                    act = real_out[:, :, neuron_indices].max(dim=0)[0]
                elif dim_reduction_method == 'mean':
                    act = real_out[:, :, neuron_indices].mean(dim=0)
                else:
                    raise Exception(f"Unknow dim_reduction_method {dim_reduction_method} in hook_get_act.")
                max_acts.append(act.squeeze().cpu().float().numpy().astype(np.float16))
            
            self.activations.setdefault(layer_name, []).extend(np.array(max_acts))
        return hook

    def hook_get_act_experts(self, layer_name, dim_reduction_method):
        def get_expert_id(name):
            parts = name.split('.')
            if self.model_config.model_name == 'gpt-oss-20b':
                return int(parts[-1])
            else:
                if len(parts) < 5 or parts[-3] != 'experts':
                    raise ValueError(f"Invalid expert layer name: {name}")
                return int(parts[-2])

        def hook(module, input, output):
            # if self.model_config.model_name != 'gpt-oss-20b':
            expert_id = get_expert_id(layer_name)
            # Retrieve token and mask maps (lists from previous hook)
            token_ids = self.expert_to_token_map.get(expert_id, [])
            mask_flags = self.expert_to_mask_map.get(expert_id, [])
            # token_weights = self.expert_to_weight_map.get(expert_id, [])
            if self.model_config.model_name == 'gpt-oss-20b':
                mask_per_token = self.mask_per_token_per_expert.get(expert_id, [])
                out = torch.cat(output, dim=-1)[mask_per_token].clone().detach()
            else:
                out = output.clone().detach()
                if self.model_config.model_name == "Hunyuan-A13B-Instruct":
                    out = out[0, :len(token_ids)]
            
            # Filter valid tokens (mask_flags == 1) directly in tensor
            valid_token_ids = token_ids[mask_flags]
            # valid_token_weights = token_weights[mask_flags]
            expert_outputs = out[mask_flags]  # Shape: [num_valid_tokens, hidden_dim]

            # Group activations by batch ID
            prompt_acts = defaultdict(list)
            for i, batch_id in enumerate(valid_token_ids):
                # prompt_acts[int(batch_id)].append(expert_outputs[i]*valid_token_weights[i])
                prompt_acts[int(batch_id)].append(expert_outputs[i])
            
            # Compute max activation per prompt
            hidden_dim = out.shape[-1]
            max_acts = []
            for batch_id in range(self.current_batch_size):
                acts = prompt_acts.get(batch_id, [])
                if acts:
                    stacked = torch.stack(acts, dim=0)
                    if dim_reduction_method == 'max':
                        reduced_act = stacked.max(dim=0)[0]
                    elif dim_reduction_method == 'mean':
                        reduced_act = stacked.mean(dim=0)
                    else:
                        raise Exception(f"Unknown dimension reduction method ({dim_reduction_method})")
                    max_act = reduced_act.cpu().float().numpy().astype(np.float16)
                else:
                    max_act = np.zeros(hidden_dim, dtype=np.float16)
                max_acts.append(max_act)

            max_acts = np.array(max_acts)
            self.activations.setdefault(layer_name, []).extend(max_acts)
            if len(self.activations[layer_name]) != self.cnt:
                raise RuntimeError(f"Unaligned size on {layer_name}: {len(self.activations[layer_name])} vs expected {self.cnt}")
        return hook
    
    def hook_get_dispatch_map(self, layer_name, target_expert_ids, gateboost_value=0):
        def hook(module, input, output):
            # Handle boosting
            if self.model_config.model_name == "deepseek-moe-16b-chat":
                out = output[2]  # shape: [batch_size * seq_len, n_experts]
            else:
                out = output

            if len(target_expert_ids) > 0 and gateboost_value != 0:
                out = out.clone()
                out[..., target_expert_ids] += gateboost_value

            # Build expert-to-token map
            self.expert_to_token_map = {eid: [] for eid in target_expert_ids}
            self.expert_to_mask_map = {eid: [] for eid in target_expert_ids}
            # self.expert_to_weight_map = {eid: [] for eid in target_expert_ids}
            if self.model_config.model_name == 'gpt-oss-20b':
                self.mask_per_token_per_expert = {eid: [] for eid in target_expert_ids}
            
            # Capacity management for Hunyuan
            if self.model_config.model_name == "Hunyuan-A13B-Instruct":
                # Extract expert usage from dispatch_mask
                _, combine_weights, dispatch_mask, _ = hunyuan_topkgating(out, self.model_config.topk)
                # Create attention mask aligned with tokens
                token_mask = self.attention_mask.flatten().to(torch.bool).to(out.device)
                batch_indices = torch.arange(self.current_batch_size, device=out.device).repeat_interleave(self.current_seq_len)
                # expert_weights = combine_weights.sum(dim=-1).to(out.device)
                for eid in target_expert_ids:
                    # Find tokens routed to expert eid at any capacity slot
                    expert_dispatch = dispatch_mask[:, eid, :]  # shape: [tokens, capacity]
                    token_dispatched = expert_dispatch.sum(dim=-1) > 0  # shape: [tokens]
                    token_indices = token_dispatched.nonzero(as_tuple=True)[0]
                    # expert_mask = dispatch_mask[:, eid]
                    # token_indices = expert_mask.nonzero(as_tuple=True)[0].tolist()
                    self.expert_to_token_map[eid] = batch_indices[token_indices]
                    self.expert_to_mask_map[eid] = token_mask[token_indices]
                    # self.expert_to_weight_map[eid] = expert_weights[token_indices, eid] 
                return out
            else:
                # Compute top-k expert indices
                if self.model_config.model_name in ["Mixtral-8x7B-Instruct-v0.1", "Qwen3-30B-A3B", "Phi-3.5-MoE-instruct", "Qwen1.5-MoE-A2.7B-Chat", "pangu-pro-moe-model"]:
                    score = out.softmax(dim=-1, dtype=torch.float)
                else:
                    score = out.softmax(dim=-1)
                # score = out.softmax(dim=-1, dtype=torch.float if self.model_config.model_name in ["Mixtral-8x7B-Instruct-v0.1", "Qwen3-30B-A3B", "Phi-3.5-MoE-instruct", "Qwen1.5-MoE-A2.7B-Chat"] else out.dtype)

                if self.model_config.model_name == "pangu-pro-moe-model":
                    num_groups = 8
                    experts_per_group = 8
                    routing_weights, selected_experts = torch.max(score.view(score.shape[0], num_groups, -1), dim = -1)
                    bias = torch.arange(0, self.model_config.num_router_expert, experts_per_group, device=routing_weights.device, dtype=torch.int64).unsqueeze(0)
                    topk_indices = selected_experts + bias
                    # we cast back to the input dtype
                    topk_weight = routing_weights.to(out.dtype)
                else:
                    topk_weight, topk_indices = score.topk(self.model_config.topk, dim=-1, sorted=self.model_config.model_name != "deepseek-moe-16b-chat")
                
                # Create attention mask
                token_mask = self.attention_mask.flatten().unsqueeze(-1).expand_as(topk_indices).to(torch.bool).to(out.device)  # same shape as topk_indices

                # Generate batch indices
                batch_indices = torch.arange(self.current_batch_size, device=out.device) \
                                    .view(-1, 1, 1) \
                                    .expand(-1, self.current_seq_len, self.model_config.topk) \
                                    .reshape(-1, self.model_config.topk).to(out.device)
                
                # Flatten
                flat_expert_ids = topk_indices.flatten()
                # flat_expert_weights = topk_weight.flatten()
                flat_batch_ids = batch_indices.flatten()
                flat_mask = token_mask.flatten()

                # Mask for target_expert_ids directly via tensor ops
                target_mask = torch.isin(flat_expert_ids, torch.tensor(target_expert_ids).to(out.device))
                selected_expert_ids = flat_expert_ids[target_mask]
                # selected_expert_weights = flat_expert_weights[target_mask]
                selected_batch_ids = flat_batch_ids[target_mask]
                selected_masks = flat_mask[target_mask]

                # Group by expert_id
                for eid in target_expert_ids:
                    expert_mask = selected_expert_ids == eid
                    self.expert_to_token_map[eid] = selected_batch_ids[expert_mask]
                    self.expert_to_mask_map[eid] = selected_masks[expert_mask]
                    # self.expert_to_weight_map[eid] = selected_expert_weights[expert_mask]
                    if self.model_config.model_name == 'gpt-oss-20b':
                        self.mask_per_token_per_expert[eid] = (topk_indices == eid).any(dim=1)
                        
                if self.model_config.model_name == "deepseek-moe-16b-chat":
                    return topk_indices, topk_weight, None   
                else:
                    return out

        return hook

    def hook_get_router_act_cnt(self, layer_name):
        def hook(module, input, output):
            #for deepseek-moe-16b-chat, the mlp.gate output is: (selected_expert_index [6,], gate_output_logit [6,], full_gate_output_logit [64,])
            if self.model_config.model_name=="deepseek-moe-16b-chat": 
                output = output[2]
            # The output is flattened in a shape (batch*seq, n_experts).
            # We need to dynamically reshape it based on the input_ids from the input tokens outputed by the tokenizer.
            output = output.view(self.current_batch_size, self.current_seq_len, output.shape[1])
            # Accroding to the Qwen3-30B-A3B, the number of activate experts is 8.
            # Compute top-k expert indices
            if self.model_config.model_name in ["Mixtral-8x7B-Instruct-v0.1", "Qwen3-30B-A3B"]:
                score = output.softmax(dim=-1, dtype=torch.float)
            else:
                score = output.softmax(dim=-1)
            
            if self.model_config.model_name == "deepseek-moe-16b-chat":
                _, topk_indices = score.topk(self.model_config.topk, dim=-1, sorted=False)
            else:
                _, topk_indices = score.topk(self.model_config.topk, dim=-1, sorted=True)
                
            # Count expert usage: output is (batch, 128)
            counts = torch.zeros((self.current_batch_size, self.model_config.num_router_expert), dtype=torch.float32, device=output.device)
            # Iterate through each item in the batch
            for b in range(self.current_batch_size):
                # Get expert indices for the current batch item: (seq_len, 8)
                experts_for_current_batch = topk_indices[b] 
                # Get the attention mask for the current batch item: (seq_len,)
                # Unsqueeze and expand to match the experts_for_current_batch shape (seq_len, 8)
                mask_for_current_batch = self.attention_mask[b].unsqueeze(-1).expand_as(experts_for_current_batch).to(experts_for_current_batch.device)
                # Filter out expert indices corresponding to padding tokens (where mask is 0)
                # This results in a 1D tensor containing only the expert IDs from actual (non-padding) tokens.
                actual_experts_in_batch = experts_for_current_batch[mask_for_current_batch == 1] 
                # Count occurrences of each expert ID for this batch item using scatter_add_
                # Only perform scatter_add_ if there are actual (non-padding) tokens in the sequence
                if actual_experts_in_batch.numel() > 0:
                    counts[b].scatter_add_(0, actual_experts_in_batch, torch.ones_like(actual_experts_in_batch, dtype=counts.dtype))
                    num_real_tokens = self.attention_mask[b].sum().item()
                    counts[b] = counts[b] / num_real_tokens
                    
            # Move the final counts to CPU and convert to NumPy array for storage/viewing
            counts_np = counts.cpu().numpy()
            self.activations.setdefault(layer_name, []).extend(counts_np)
        return hook

    def register_hooks(self, hook_type, target_layers, target_neuron_dict=None, patcher=None, prune_value=0, alpha=1, dim_reduction_method=None, use_softmax=False, gateboost_value=0, benign_gate_weight=None, verbose=True):
        """
        Registers forward hooks on the target modules based on candidate neurons.
        Only layers whose names contain a keyword (formatted as ".{keyword}.mlp") 
        matching one of the target layers are monitored.
        """
        self.hook_handles = []
        # Helper for registering a hook
        def _register_hook(module, hook_fn, name):
            handle = module.register_forward_hook(hook_fn)
            self.hook_handles.append(handle)
            if verbose:
                print(f"Registered {hook_type} hook on: {name}")
        
        def _get_value_from_dict(dictionary, key, info="sn indice"):
            value = dictionary.get(key, [])
            if len(value)==0:
                print(f"Warning: {value} is obtained from the key {key} (in {hook_type} for {info})")
            return value
        
        module_dict = dict(self.model.named_modules())
        if target_neuron_dict is None:
            if hook_type in ["get_act_expert", "prune_sn"]:
                raise Exception(f"safety_neuron_dict cannot be None for {hook_type} type.")
        
        dispatcher_registered = False
        expert_registered = False
        self.shared_weight = None
        for layer_name in target_layers:
            module = module_dict.get(layer_name)
            if module is None:
                print(f"Error: Could not find module for layer {layer_name}!!!")
                continue

            if hook_type == "get_act":
                if dim_reduction_method is None:
                    raise Exception(f"dim_reduction_method cannot be None for the {hook_type} hook.")
                _register_hook(module, self.hook_get_act(layer_name, dim_reduction_method), layer_name)
            elif hook_type == "get_act_router_cnt":
                _register_hook(module, self.hook_get_router_act_cnt(layer_name), layer_name)

            elif hook_type == "get_act_expert":
                if dim_reduction_method is None:
                    raise Exception(f"dim_reduction_method cannot be None for the {hook_type} hook.")
                if layer_name.endswith(self.model_config.name_gate):
                    neuron_indices = _get_value_from_dict(target_neuron_dict, layer_name)
                    _register_hook(module, self.hook_get_dispatch_map(layer_name, neuron_indices, gateboost_value=gateboost_value), layer_name)
                    dispatcher_registered = True
                elif self.model_config.name_router_expert+'.' in layer_name:
                    _register_hook(module, self.hook_get_act_experts(layer_name, dim_reduction_method), layer_name)
                    expert_registered = True

            elif hook_type == "get_act_shared_expert":
                if layer_name.endswith(self.model_config.name_gate):
                    neuron_indices = _get_value_from_dict(target_neuron_dict, layer_name)
                    _register_hook(module, self.hook_prune(layer_name, neuron_indices, prune_value=prune_value), layer_name)
                elif self.model_config.name_shared_expert+'.' in layer_name:
                    if dim_reduction_method is None:
                        raise Exception(f"dim_reduction_method cannot be None for the {hook_type} hook.")
                    _register_hook(module, self.hook_get_act(layer_name, dim_reduction_method), layer_name)
                if self.model_config.model_name == "deepseek-moe-16b-chat":
                    if layer_name in ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"]:
                        if dim_reduction_method is None:
                            raise Exception(f"dim_reduction_method cannot be None for the {hook_type} hook.")
                        _register_hook(module, self.hook_get_act(layer_name, dim_reduction_method), layer_name) 
            
            elif hook_type == "prune_sn":
                neuron_indices = _get_value_from_dict(target_neuron_dict, layer_name)
                _register_hook(module, self.hook_prune(layer_name, neuron_indices, prune_value=prune_value), layer_name)

            else:
                raise Exception(f"Unknown hook_type ({hook_type})")
            
        if hook_type == "get_act_expert":
            if not dispatcher_registered or not expert_registered:
                raise Exception(f"Both dispatcher hooks (register_stat={dispatcher_registered}) and expert activation hooks (register_stat={expert_registered}) must be registered for {hook_type} type.")

    def prepare_for_batch(self, batch_size: int, seq_len: int, attention_mask: torch.Tensor):
        self.clear_collected_data() # This clears the maps *before* a new batch starts
        self.current_batch_size = batch_size
        self.current_seq_len = seq_len
        self.attention_mask = attention_mask.to(self.model.device) 

    def clear_collected_data(self):
        # self.expert_to_token_map = {}
        self.attention_mask = None

    def fill_missing_expert_activations(self, name_expert_layers):
        """
        For each expert layer, if the hook was not called for this batch,
        append zeros for each prompt to self.activations[layer_name].
        """
        for layer_name in name_expert_layers:
            if "expert" in layer_name and not layer_name.endswith("experts") and not layer_name.endswith("shared_expert_gate"):
                acts = self.activations.get(layer_name, [])
                # If the hook was called, acts will have >= self.cnt entries for this batch
                # If not, acts will not be updated for this batch
                # We append zeros for this batch
                if len(acts) < self.cnt:
                    print(f"padding zeros in self.activations for {layer_name}")
                    num_to_add = self.cnt - len(acts)
                    hidden_size = self.get_module_by_name(self.model, layer_name).out_features
                    self.activations.setdefault(layer_name, []).extend(np.zeros((num_to_add, hidden_size), dtype=np.float32))

    def get_question_mask(self, questions, ids_w_template):
        """
        Returns a boolean mask for the prompt tokens in each chat template in the batch.
        input_texts: list of prompt strings
        Returns: np.ndarray of shape (batch_size, seq_len), True for prompt tokens, False otherwise
        """
        masks = torch.zeros_like(ids_w_template, dtype=bool)
        for i, id_w_template in enumerate(ids_w_template):
            id_wo_template = self.tokenizer(questions[i], return_tensors="pt", padding=False, truncation=True)["input_ids"][0]
            len_question = len(id_wo_template)
            match_detected = False
            for j in range(masks.shape[1] - len_question + 1):
                if torch.equal(id_w_template[j+5:j+len_question], id_wo_template[5:]):
                    # We exclude the first three tokens
                    masks[i, j+3:j+len_question] = True
                    match_detected = True
                    break
            if not match_detected:
                print(id_w_template)
                print(id_wo_template)
                raise Exception("No matched tokens between queries with and without template.")

        return masks  # shape: [batch_size, seq_len]

    def inference_no_response(self, input_text, name_expert_layers=None, batch_size=32):
        """
        Resets the stored activations, runs the model on the provided prompts,
        concatenates the activations per layer, and returns both the model's responses 
        and the neuron activations.
        """
        # Reset activations before generating new output.
        self.cnt = 0
        self.activations = {}
        
        # register the activation hook everytime when calling the function
        prompts = util_model.construct_prompt(self.tokenizer, self.model_config.model_name, input_text)
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        for batch_prompts in tqdm(util.batchify(prompts, batch_size), total=total_batches):
            input_tokens = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            # Remove token_type_ids if not accepted by the model
            if "token_type_ids" in input_tokens:
                forward_args = inspect.signature(self.model.forward).parameters
                if "token_type_ids" not in forward_args:
                    input_tokens.pop("token_type_ids")
            # For the reshaping of the MoE gate hook
            current_batch_size, current_seq_len = input_tokens['input_ids'].shape
            current_questions = [input_text[i] for i in range(self.cnt, self.cnt+current_batch_size)]
            attention_mask = self.get_question_mask(current_questions, input_tokens['input_ids'].cpu())
            self.cnt += current_batch_size
            # attention_mask = input_tokens["attention_mask"]

            self.prepare_for_batch(current_batch_size, current_seq_len, attention_mask)
            self.model.eval()
            with torch.no_grad():
                _ = self.model(**input_tokens)
            if name_expert_layers is not None:
                self.fill_missing_expert_activations(name_expert_layers)

        # clear the hooks
        if hasattr(self, 'hook_handles'):
            self.remove_hooks()
        return self.activations
    
    def inference(self, input_text, batch_size=24, max_new_tokens=1024, enable_thinking=False):
        """
        Resets the stored activations, runs the model on the provided prompts,
        concatenates the activations per layer, and returns both the model's responses 
        and the neuron activations.
        """
        # Reset activations before generating new output.
        self.activations = {}
        prompts = util_model.construct_prompt(self.tokenizer, self.model_config.model_name, input_text, enable_thinking=enable_thinking)
        responses = util_model.generate_output(
            self.model, self.tokenizer, prompts, 
            batch_size=batch_size, max_new_tokens=max_new_tokens, model_name=self.model_config.model_name
        )
        if self.activations:
            # Concatenate activations for each layer.
            for layer_name in self.activations:
                self.activations[layer_name] = np.concatenate(self.activations[layer_name], axis=0)
        
        # clear the hooks
        if hasattr(self, 'hook_handles'):
            self.remove_hooks()
        return responses

    def remove_hooks(self):
        """
        Removes all the forward hooks. Call this when you no longer need to monitor activations.
        """
        if self.hook_handles:
            for hook in self.hook_handles:
                hook.remove()