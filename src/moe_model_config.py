from dataclasses import dataclass
from typing import Optional

@dataclass
class MoEModelConfig:
    model_name: str
    name_gate: str
    topk: int
    num_router_expert: int
    name_router_expert: str
    name_expert_layers: str
    name_shared_expert_gate: Optional[str] = 'None'
    name_shared_expert: Optional[str] = 'None'

models = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": MoEModelConfig(
        model_name="Qwen3-30B-A3B-Instruct-2507", 
        name_gate="mlp.gate", 
        topk=8,
        num_router_expert=128,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_proj", "up_proj"]),
    "Qwen/Qwen3-30B-A3B-Thinking-2507": MoEModelConfig(
        model_name="Qwen3-30B-A3B-Thinking-2507", 
        name_gate="mlp.gate", 
        topk=8,
        num_router_expert=128,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_proj", "up_proj"]),
    "Qwen/Qwen3-30B-A3B": MoEModelConfig(
        model_name="Qwen3-30B-A3B", 
        name_gate="mlp.gate", 
        topk=8,
        num_router_expert=128,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_proj", "up_proj"]),
    "microsoft/Phi-3.5-MoE-instruct": MoEModelConfig(
        model_name="Phi-3.5-MoE-instruct", 
        name_gate="block_sparse_moe.gate", 
        topk=2,
        num_router_expert=16,
        name_router_expert="block_sparse_moe.experts",
        name_expert_layers=["w1", "w3"]),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": MoEModelConfig(
        model_name="Mixtral-8x7B-Instruct-v0.1", 
        name_gate="block_sparse_moe.gate", 
        topk=2, 
        num_router_expert=8,
        name_router_expert="block_sparse_moe.experts",
        name_expert_layers=["w1", "w3"]),
    "openai/gpt-oss-20b": MoEModelConfig(
        model_name="gpt-oss-20b", 
        name_gate="mlp.router", 
        topk=4, 
        num_router_expert=32,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_up_layers"]),
    "Qwen/Qwen1.5-MoE-A2.7B-Chat": MoEModelConfig(
        model_name="Qwen1.5-MoE-A2.7B-Chat", 
        name_gate="mlp.gate", # and mlp.shared_expert_gate for the (4) shared experts
        topk=4, 
        num_router_expert=60,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert_gate="mlp.shared_expert_gate",
        name_shared_expert="mlp.shared_expert"),
    "tencent/Hunyuan-A13B-Instruct": MoEModelConfig(
        model_name="Hunyuan-A13B-Instruct", 
        name_gate="mlp.gate.wg",
        topk=8, 
        num_router_expert=64,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert="mlp.shared_mlp"),
    "deepseek-ai/deepseek-moe-16b-chat": MoEModelConfig(
        model_name="deepseek-moe-16b-chat", 
        name_gate="mlp.gate", 
        topk=6, 
        num_router_expert=64,
        name_router_expert="mlp.experts", # the first layer only have shared expert
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert="mlp.shared_experts"),
    "IntervitensInc/pangu-pro-moe-model": MoEModelConfig(
        model_name="pangu-pro-moe-model", 
        name_gate="mlp.gate", 
        topk=8, 
        num_router_expert=64,
        name_router_expert="mlp.experts",
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert="mlp.shared_expert"),
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": MoEModelConfig(
        model_name="Llama-4-Scout-17B-16E-Instruct", 
        name_gate="feed_forward.router", 
        topk=1, 
        num_router_expert=16,
        name_router_expert="feed_forward.experts",
        name_expert_layers=["gate_proj", "up_proj"],
        name_shared_expert="feed_forward.shared_expert"),
}
