import torch.nn.functional as F
import torch
import torch.nn as nn

def deepseek_moe_gate_forward(self, hidden_states):
    bsz, seq_len, h = hidden_states.shape        
    ### compute gating score
    hidden_states = hidden_states.view(-1, h)
    logits = F.linear(hidden_states, self.weight, None)
    if self.scoring_func == 'softmax':
        scores = logits.softmax(dim=-1)
    else:
        raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
    
    ### select top-k experts
    topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
    
    ### norm gate to sum 1
    if self.top_k > 1 and self.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator

    ### expert-level computation auxiliary loss
    if self.training and self.alpha > 0.0:
        scores_for_aux = scores
        aux_topk = self.top_k
        # always compute aux loss based on the naive greedy topk method
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        if self.seq_aux:
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
            ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
        else:
            mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (Pi * fi).sum() * self.alpha
    else:
        aux_loss = logits
    return topk_idx, topk_weight, aux_loss

def kimi_vl_moe_gate_forward(self, hidden_states):
    bsz, seq_len, h = hidden_states.shape
    # compute gating score
    hidden_states = hidden_states.view(-1, h)
    logits = F.linear(
        hidden_states.type(torch.float32), self.weight.type(torch.float32), None
    )
    if self.scoring_func == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise NotImplementedError(
            f"insupportable scoring function for MoE gating: {self.scoring_func}"
        )

    # select top-k experts
    if self.topk_method == "noaux_tc":
        assert not self.training
        scores_for_choice = scores.view(
            bsz * seq_len, -1
        ) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )  # [n, n_group]
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
            )
            .reshape(bsz * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(
            ~score_mask.bool(), 0.0
        )  # [n, e]
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
    elif self.topk_method == "greedy":
        topk_weight, topk_idx = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )
    else:
        raise NotImplementedError(
            f"insupportable TopK function for MoE gating: {self.topk_method}"
        )

    # norm gate to sum 1
    if self.top_k > 1 and self.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
    topk_weight = (
        topk_weight * self.routed_scaling_factor
    )  # must multiply the scaling factor

    if self.training and self.alpha > 0.0:
        scores_for_aux = scores
        aux_topk = self.top_k
        # always compute aux loss based on the naive greedy topk method
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        if self.seq_aux:
            scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
            ce = torch.zeros(
                bsz, self.n_routed_experts, device=hidden_states.device
            )
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
            ).div_(seq_len * aux_topk / self.n_routed_experts)
            aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                dim=1
            ).mean() * self.alpha
        else:
            mask_ce = F.one_hot(
                topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
            )
            ce = mask_ce.float().mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (Pi * fi).sum() * self.alpha
    else:
        aux_loss = logits

    return topk_idx, topk_weight, aux_loss

def hunyuan_topkgating(logits: torch.Tensor, topk: int):
    logits = logits.float()
    gates = F.softmax(logits, dim=1)
    # expert_capacity = topk * gates.shape[0]
    expert_capacity = max(topk, topk * gates.shape[0] // gates.shape[1])
    # expert_capacity = int(1e9)
    num_experts = int(gates.shape[1])
    # Top-k router probability and corresponding expert indices for each token.
    # Shape: [tokens_per_group, num_selected_experts].
    expert_gate, expert_index = torch.topk(gates, topk)
    expert_mask = F.one_hot(expert_index, num_experts)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [tokens_per_group, num_experts]
    expert_mask_aux = expert_mask.max(dim=-2)[0]
    tokens_per_group_and_expert = torch.mean(expert_mask_aux.float(), dim=-2)
    router_prob_per_group_and_expert = torch.mean(gates.float(), dim=-2)
    l_aux = num_experts**2 * torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert)

    gates_s = torch.clamp(
        torch.matmul(expert_mask.float(), gates.unsqueeze(-1)).sum(dim=1), min=torch.finfo(gates.dtype).eps
    )
    router_probs = gates / gates_s
    # Make num_selected_experts the leading axis to ensure that top-1 choices
    # have priority over top-2 choices, which have priority over top-3 choices,
    # etc.
    expert_index = torch.transpose(expert_index, 0, 1)
    # Shape: [num_selected_experts * tokens_per_group]
    expert_index = expert_index.reshape(-1)

    # Create mask out of indices.
    # Shape: [tokens_per_group * num_selected_experts, num_experts].
    expert_mask = F.one_hot(expert_index, num_experts).to(torch.int32)
    exp_counts = torch.sum(expert_mask, dim=0).detach()

    # Experts have a fixed capacity that we cannot exceed. A token's priority
    # within the expert's buffer is given by the masked, cumulative capacity of
    # its target expert.
    # Shape: [tokens_per_group * num_selected_experts, num_experts].
    token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask - 1
    # Shape: [num_selected_experts, tokens_per_group, num_experts].
    token_priority = token_priority.reshape((topk, -1, num_experts))
    # Shape: [tokens_per_group, num_selected_experts, num_experts].
    token_priority = torch.transpose(token_priority, 0, 1)
    # For each token, across all selected experts, select the only non-negative
    # (unmasked) priority. Now, for group G routing to expert E, token T has
    # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
    # is its targeted expert.
    # Shape: [tokens_per_group, num_experts].
    token_priority = torch.max(token_priority, dim=1)[0]

    # Token T can only be routed to expert E if its priority is positive and
    # less than the expert capacity. One-hot matrix will ignore indices outside
    # the range [0, expert_capacity).
    # Shape: [tokens_per_group, num_experts, expert_capacity].
    valid_mask = torch.logical_and(token_priority >= 0, token_priority < expert_capacity)
    token_priority = torch.masked_fill(token_priority, ~valid_mask, 0)
    dispatch_mask = F.one_hot(token_priority, expert_capacity).to(torch.bool)
    valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, expert_capacity)
    dispatch_mask = torch.masked_fill(dispatch_mask, ~valid_mask, 0)

    # The combine array will be used for combining expert outputs, scaled by the
    # router probabilities. Shape: [num_groups, tokens_per_group, num_experts,
    # expert_capacity].
    combine_weights = torch.einsum("...te,...tec->...tec", router_probs, dispatch_mask)
    exp_counts_capacity = torch.sum(dispatch_mask)
    exp_capacity_rate = exp_counts_capacity / (logits.shape[0]*topk)

    return [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts

def qwen1_5_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """ """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)
    shared_expert_logits = self.shared_expert_gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hitted:
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    shared_expert_output = self.shared_expert(hidden_states)
    shared_expert_output = F.sigmoid(shared_expert_logits) * shared_expert_output

    final_hidden_states = final_hidden_states + shared_expert_output

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def GptOssTopKRouter_forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
    return router_logits

def GptOssMLP_forward(self, hidden_states):
    router_logits = self.router(hidden_states)  # (num_experts, seq_len)
    # We hardcode the topk expert (4) for the 20b model; change it for bigger one
    router_top_value, router_indices = torch.topk(router_logits, 4, dim=-1)  # (seq_len, top_k)
    router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
    router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
    routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
    return routed_out, router_scores


class GptOssGateUpLayer(nn.Module):
    def __init__(self, hidden_size, expert_dim):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(hidden_size, 2 * expert_dim))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(2 * expert_dim))

    def forward(self, hidden_states):
        gate_up = hidden_states @ self.gate_up_proj + self.gate_up_proj_bias
        return gate_up[..., ::2], gate_up[..., 1::2]

class GptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        # One GptOssGateUpLayer per expert
        self.gate_up_layers = nn.ModuleList([
            GptOssGateUpLayer(self.hidden_size, self.expert_dim)
            for _ in range(self.num_experts)
        ])

        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.expert_dim, self.hidden_size))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]

        if self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit[:]:
                expert_id = expert_idx[0].item()
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_id])

                current_state = hidden_states[token_idx]
                gate, up = self.gate_up_layers[expert_id](current_state)

                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu

                out = gated_output @ self.down_proj[expert_id] + self.down_proj_bias[expert_id]
                weighted_output = out[0] * routing_weights[token_idx, expert_id, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

            next_states = next_states.view(batch_size, -1, self.hidden_size)

        else:
            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)

            gate, up = [], []
            for i in range(num_experts):
                g, u = self.gate_up_layers[i](hidden_states[i])
                gate.append(g)
                up.append(u)
            gate = torch.stack(gate, dim=0)
            up = torch.stack(up, dim=0)

            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)

            next_states = torch.bmm((up + 1) * glu, self.down_proj)
            next_states = next_states + self.down_proj_bias[..., None, :]
            next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
            next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
            next_states = next_states.sum(dim=0)

        return next_states