import nltk
from A1_skeleton import build_tokenizer

nltk.download("punkt")
nltk.download("punkt_tab")

tokenizer = build_tokenizer(
    train_file="train.txt",
    max_voc_size=10000,
    model_max_length=128,
)

print("Vocabulary size:", len(tokenizer))
print("PAD:", tokenizer.pad_token_id, tokenizer.pad_token)
print("UNK:", tokenizer.unk_token_id, tokenizer.unk_token)
print("BOS:", tokenizer.bos_token_id, tokenizer.bos_token)
print("EOS:", tokenizer.eos_token_id, tokenizer.eos_token)

out = tokenizer(
    ["This is a test.", "Another test."],
    truncation=True,
    padding=True,
    return_tensors="pt",
)

print(out)
print(out["input_ids"].shape)
print(out["attention_mask"].shape)

for row in out["input_ids"]:
    print([tokenizer.int_to_str[int(i)] for i in row])

tokenizer.save("a1_tokenizer.pkl")
print("Saved tokenizer.")

print("'the' in vocab:", "the" in tokenizer.str_to_int)
print("'and' in vocab:", "and" in tokenizer.str_to_int)
print("'cuboidal' in vocab:", "cuboidal" in tokenizer.str_to_int)
print("'epiglottis' in vocab:", "epiglottis" in tokenizer.str_to_int)

word = "the"
idx = tokenizer.str_to_int[word]
word_back = tokenizer.int_to_str[idx]

print(word, "->", idx, "->", word_back)