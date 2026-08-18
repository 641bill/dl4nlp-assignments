import torch

from A2_skeleton import A2ModelConfig, A2Transformer
from generate_a2 import generate_text, predict_next_word


class TinyTokenizer:
    pad_token_id = 0
    unk_token_id = 1
    bos_token_id = 2
    eos_token_id = 3
    int_to_str = {i: f"t{i}" for i in range(32)}
    int_to_str[2] = "<BOS>"
    int_to_str[3] = "<EOS>"
    str_to_int = {s: i for i, s in int_to_str.items()}

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        ids = [[self.bos_token_id, 4, 5, self.eos_token_id]]
        if return_tensors == "pt":
            ids = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": ids}


config = A2ModelConfig(
    vocab_size=32,
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_hidden_layers=1,
    rope_theta=10000.0,
    hidden_act="silu",
    max_position_embeddings=32,
    rms_norm_eps=1e-6,
)
model = A2Transformer(config)
model.eval()
tokenizer = TinyTokenizer()

word, ranked = predict_next_word(model, tokenizer, "dummy", k=3, device="cpu")
assert isinstance(word, str)
assert len(ranked) == 3

torch.manual_seed(0)
greedy = generate_text(
    model, tokenizer, "dummy", max_length=8, temperature=0.0, topk=None, device="cpu"
)
assert isinstance(greedy, str)
assert len(greedy.split()) >= 1

torch.manual_seed(0)
sampled = generate_text(
    model, tokenizer, "dummy", max_length=8, temperature=1.0, topk=5, device="cpu"
)
assert isinstance(sampled, str)

print("greedy:", greedy)
print("sampled:", sampled)
print("Task 3.2 smoke test passed!")
