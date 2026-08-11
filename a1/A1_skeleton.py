import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os

###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """

    # TODO: build the vocabulary, possibly truncating it to max_voc_size if that is specified.
    # Then return a tokenizer object (implemented below).
    counter = Counter()

    with open(train_file, encoding='utf-8') as f:
        for line in f: # Read train file line by line.
            line = line.strip()
            if line:
                tokens = tokenize_fun(line) # Split each paragraph into lowercase tokens.
                counter.update(tokens) # Count the frequency of each token.
    
    special_tokens = [pad_token, unk_token, bos_token, eos_token]
    str_to_int = {}

    # Add special tokens first.
    for token in special_tokens:
        if token in str_to_int:
            raise ValueError(f'Duplicate special token: {token}')
        str_to_int[token] = len(str_to_int)

    # max_voc_size includes the special tokens.
    if max_voc_size is None:
        most_common = counter.most_common()
    else:
        remaining_slots = max_voc_size - len(special_tokens)
        if remaining_slots < 0:
            raise ValueError('max_voc_size must be at least the number of special tokens.')
        most_common = counter.most_common(remaining_slots)

    for token, _ in most_common:
        if token not in str_to_int:
            str_to_int[token] = len(str_to_int)

    int_to_str = {i: s for s, i in str_to_int.items()}

    return A1Tokenizer(
        str_to_int=str_to_int,
        int_to_str=int_to_str,
        tokenize_fun=tokenize_fun,
        model_max_length=model_max_length,
        pad_token=pad_token,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
    )
    
class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, str_to_int, int_to_str, tokenize_fun, model_max_length,
                 pad_token, unk_token, bos_token, eos_token):
        # TODO: store all values you need in order to implement __call__ below.
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str
        self.tokenize_fun = tokenize_fun

        self.model_max_length = model_max_length

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.pad_token_id = self.str_to_int[self.pad_token]
        self.unk_token_id = self.str_to_int[self.unk_token]
        self.bos_token_id = self.str_to_int[self.bos_token]
        self.eos_token_id = self.str_to_int[self.eos_token]

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens.
           
           Args:
             texts:           The texts to tokenize.
             truncation:      Whether the texts should be truncated to model_max_length.
             padding:         Whether the tokenized texts should be padded on the right side.
             return_tensors:  If None, then return lists; if 'pt', then return PyTorch tensors.

           Returns:
             A BatchEncoding where the field `input_ids` stores the integer-encoded texts.
        """
        # TODO: Your work here is to split the texts into words and map them to integer values.
        # 
        # - If `truncation` is set to True, the length of the encoded sequences should be 
        #   at most self.model_max_length.
        # - If `padding` is set to True, then all the integer-encoded sequences should be of the
        #   same length. That is: the shorter sequences should be "padded" by adding dummy padding
        #   tokens on the right side.
        # - If `return_tensors` is undefined, then the returned `input_ids` should be a list of lists.
        #   Otherwise, if `return_tensors` is 'pt', then `input_ids` should be a PyTorch 2D tensor.

        # TODO: Return a BatchEncoding where input_ids stores the result of the integer encoding.
        # Optionally, if you want to be 100% HuggingFace-compatible, you should also include an 
        # attention mask of the same shape as input_ids. In this mask, padding tokens correspond
        # to the the value 0 and real tokens to the value 1.
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
            
        encoded_texts = []

        for text in texts:
            tokens = self.tokenize_fun(text)

            if truncation and self.model_max_length is not None:
                max_content_length = self.model_max_length - 2
                tokens = tokens[:max_content_length]

            ids = [self.bos_token_id]
            ids += [
                self.str_to_int.get(token, self.unk_token_id)
                for token in tokens
            ]
            ids += [self.eos_token_id]

            encoded_texts.append(ids)

        attention_mask = [[1] * len(ids) for ids in encoded_texts]

        if padding:
            max_len = max(len(ids) for ids in encoded_texts)

            padded_texts = []
            padded_attention_masks = []

            for ids, mask in zip(encoded_texts, attention_mask):
                ids = ids[:max_len]
                mask = mask[:max_len]

                num_padding = max_len - len(ids)

                padded_texts.append(ids + [self.pad_token_id] * num_padding)
                padded_attention_masks.append(mask + [0] * num_padding)

            encoded_texts = padded_texts
            attention_mask = padded_attention_masks

        if return_tensors == 'pt':
            encoded_texts = torch.tensor(encoded_texts, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return BatchEncoding({
            'input_ids': encoded_texts,
            'attention_mask': attention_mask,
        })

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=2000, embedding_size=32, hidden_size=64, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_size
        )
        self.rnn = nn.LSTM(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            batch_first=True
        )
        self.unembedding = nn.Linear(
            config.hidden_size,
            config.vocab_size
        )

        # Note: -100 is the value HuggingFace conventionally uses to refer to tokens
        # where we do not want to compute the loss.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        self.post_init()

    def forward(self, input_ids, labels=None):
        """The forward pass of the RNN-based language model.
        
           Args:
             - input_ids:  The input tensor (2D), consisting of a batch of integer-encoded texts.
             - labels:     The reference tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             A CausalLMOutput containing
               - logits:   The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
               - loss:     The loss computed on this batch.               
        """
        embedded = self.embedding(input_ids)
        rnn_out, _ = self.rnn(embedded)
        logits = self.unembedding(rnn_out)
        loss = None
        # Task 3.2
        if labels is not None:
            shifted_logits = logits[:, :-1, :] # (batch, seq_len-1, vocab_size)
            shifted_labels = labels[:, 1:] # (batch, seq_len-1)
            shifted_logits = shifted_logits.reshape(-1, shifted_logits.shape[-1]) # 3-dimensional -> 2-dimensional
            shifted_labels = shifted_labels.reshape(-1) # 2-dimensional -> 1-dimensional
            loss = self.loss_func(shifted_logits, shifted_labels)

        return CausalLMOutput(logits=logits, loss=loss)


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        # loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate
        )

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False
        )
        
        # TODO: Your work here is to implement the training loop.
        #       
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       input_ids = apply your tokenizer to B
        #       labels = input_ids with padding replaced by -100
	    #       put input_ids and labels onto the GPU (or whatever device you use)
        #       apply the model to input_ids and labels
        #       get the loss from the model output
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward()
        #       optimizer.step()
        for epoch in range(int(args.num_train_epochs)):
            self.model.train()
            total_train_loss = 0.0

            for batch in train_loader:
                encoding = self.tokenizer(
                    batch["text"],
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )

                input_ids = encoding["input_ids"]
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                output = self.model(
                    input_ids=input_ids,
                    labels=labels
                )
                loss = output.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            print(
                f"Epoch {epoch + 1}/{int(args.num_train_epochs)} "
                f"- train loss: {avg_train_loss:.4f}"
            )
            # -------------------------
            # Validation
            # -------------------------
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    encoding = self.tokenizer(
                        batch["text"],
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                    )

                    input_ids = encoding["input_ids"]

                    labels = input_ids.clone()
                    labels[
                        labels == self.tokenizer.pad_token_id
                    ] = -100

                    input_ids = input_ids.to(device)
                    labels = labels.to(device)

                    output = self.model(
                        input_ids=input_ids,
                        labels=labels
                    )

                    total_val_loss += output.loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                print(
                    f'Epoch {epoch + 1}/{int(args.num_train_epochs)} '
                    f'- train loss: {avg_train_loss:.4f} '
                    f'- val loss: {avg_val_loss:.4f}'
                )

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)
