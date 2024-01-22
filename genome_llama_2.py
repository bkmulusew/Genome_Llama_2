import torch
import pandas as pd
import os
import random
import numpy as np
import pytorch_lightning as pl
from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


# Constants
DNA_VALID_NUCLEOTIDES = {'A', 'C', 'G', 'T', 'N'}
MIN_SEQUENCE_LENGTH = 100000
TRAIN_RATIO = 0.8
VALID_RATIO = 0.9
LLM_MODEL = "meta-llama/Llama-2-7b-hf"
NEW_TOKENIZER_VOCAB_SIZE = 1000
RAW_DATA_FILE_PATH = "genome_data/GRCh38_genomic.txt"
TOKENIZER_SAVE_PATH = "genome_llama2_tokenizer/Genome-Llama-2-7b-tokenizer"
CONTEXT_LENGTH = 4096


class GenomeDataPreprocessor:

    def __init__(self, file_path):
        self.file_path = file_path

    def load_fna_file(self):
        try:
            with open(self.file_path, 'r') as file:
                return [line.strip() for line in file if line.strip()]
        except IOError as e:
            print(f"Error opening file: {e}")
            return []

    def is_valid_dna_sequence(self, sequence):
        return all(nucleotide in DNA_VALID_NUCLEOTIDES for nucleotide in sequence)

    def parse_genome_file(self):
        sequences = []
        concatenated_sequence = ''

        for sequence in self.load_fna_file(self.file_path):
            sequence = sequence.upper()
            if self.is_valid_dna_sequence(sequence):
                concatenated_sequence += sequence
                while len(concatenated_sequence) >= MIN_SEQUENCE_LENGTH:
                    sequences.append(concatenated_sequence[:MIN_SEQUENCE_LENGTH])
                    concatenated_sequence = concatenated_sequence[MIN_SEQUENCE_LENGTH:]
            # else:
            #     if concatenated_sequence:
            #         sequences.append(concatenated_sequence)
            #     concatenated_sequence = ''  # Reset for new valid sequences

        # Append any remaining valid sequence
        if concatenated_sequence:
            sequences.append(concatenated_sequence)

        return sequences

    def create_dataset(self, sequences):
        df = pd.DataFrame(sequences, columns=['sequence'])
        return Dataset.from_pandas(df)

    def split_dataset(self, dataset, train_size, valid_size):
        train_dataset = dataset[:train_size]
        valid_dataset = dataset[train_size:valid_size]
        test_dataset = dataset[valid_size:]
        return train_dataset, valid_dataset, test_dataset


class GenomeLlama2TokenizerTrainer:

    def __init__(self, dataset):
        self.dataset = dataset

    def get_training_corpus(self, chunk_size=1000):
        return (self.dataset["sequence"][i : i + chunk_size] for i in range(0, len(self.dataset), chunk_size))

    def train_tokenizer(self):
        training_corpus = self.get_training_corpus()
        old_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, NEW_TOKENIZER_VOCAB_SIZE)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token
        return tokenizer


class TokenizedDataset:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def shared_transform(self, processed_dataset):
        tokenized_ds = processed_dataset.map(
            self.tokenize,
            remove_columns=processed_dataset["train"].column_names,
            batched=True,
            load_from_cache_file=True,
        )

        return tokenized_ds
    
    def create_label(self, original_list):
        flat_list = [item for sublist in original_list for item in sublist]
        flat_list.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token))
        shifted_flat_list = flat_list[1:]
        shifted_list = []
        row_length = len(original_list[0])

        for i in range(0, len(shifted_flat_list), row_length):
            shifted_list.append(shifted_flat_list[i:i + row_length])

        last_row = shifted_list[-1]
        if len(last_row) < row_length:
            last_row.extend([None] * (row_length - len(last_row)))

        return shifted_list

    def tokenize(self, element):
        outputs = self.tokenizer(
            element['sequence'],
            truncation=True,
            padding="max_length",
            max_length=CONTEXT_LENGTH,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        attention_batch = []
        for length, input_ids, attention_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
            if length == CONTEXT_LENGTH:
                input_batch.append(input_ids)
                attention_batch.append(attention_mask)
        labels_batch = self.create_label(input_batch)
        return {"input_ids": input_batch, "attention_mask": attention_batch, "labels": labels_batch}


# Preprocess Genome Data
preprocessor = GenomeDataPreprocessor(RAW_DATA_FILE_PATH)
sequences = preprocessor.parse_genome_file()
train_size = int(len(sequences) * TRAIN_RATIO)
val_size = int(len(sequences) * VALID_RATIO)
ds_train, ds_valid, ds_test = preprocessor.split_dataset(sequences, train_size, val_size)
raw_datasets = DatasetDict({
    "train": preprocessor.create_dataset(ds_train),
    "valid": preprocessor.create_dataset(ds_valid),
    "test": preprocessor.create_dataset(ds_test)
})


# Tokenizer training
trainer = GenomeLlama2TokenizerTrainer(raw_datasets['train'])
tokenizer = trainer.train_new_tokenizer()
print("Tokenizer trained successfully.")
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
print(f"Tokenizer saved successfully to {TOKENIZER_SAVE_PATH}.")


# Tokenizer usage example
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SAVE_PATH)
tokenizer.pad_token = tokenizer.eos_token
example = "AGCTTAGCTAGTCGTAGCTAATCGATCGATCGATCGTAGCTAGCTAGCTAAGCTTAGCTA"
tokens = tokenizer.tokenize(example)
print(tokens)


# Tokenized dataset
tokenizedDataset = TokenizedDataset(tokenizer)
tokenizedDataset = tokenizedDataset.shared_transform(raw_datasets)


# Arguments for Pytorch Lightning Trainer
args = {
    'epochs': 20,
    'gpus': -1,
    'lr': 1e-3,
    'precision': 16,
    'batch_size': 8,
    'num_workers': 4,
    'accumulate_grad_batches': 2,
    'enable_checkpointing': False,
    'profiler': 'advanced',
}


class GenomeLLAMA2(LightningModule):
    def __init__(self, tokenizer, tokenized_dataset, learning_rate):
        super().__init__()
        self.model = None
        self.tokenizer = tokenizer
        self.tokenized_dataset = tokenized_dataset
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask, labels):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def _transform_tensors(self, input_tensors):
        # Assuming all tensors have the same length
        num_tensors = len(input_tensors)
        tensor_length = len(input_tensors[0])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Transposing the tensors
        output_tensors = [torch.tensor([input_tensors[j][i] for j in range(num_tensors)]) for i in range(tensor_length)]

        return torch.stack(output_tensors).to(device)

    def _step(self, batch):
        input_ids = self._transform_tensors(batch['input_ids'])
        attention_mask = self._transform_tensors(batch['attention_mask'])
        labels = self._transform_tensors(batch['labels'])
        loss, _ = self(input_ids, attention_mask, labels)
        # Write the loss function
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_model(self):
        if self.model is not None:
            return
        config = AutoConfig.from_pretrained(
            LLM_MODEL,
            vocab_size=len(self.tokenizer),
            n_ctx=CONTEXT_LENGTH,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.model = LlamaForCausalLM(config)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10),
            'monitor': 'val_loss',
            'frequency': 1,
            'interval': 'epoch'
        }
        return [optimizer], [lr_scheduler]
        
    def _get_dataloader(self, split_name):
        sampler = torch.utils.data.DistributedSampler(self.tokenized_dataset[split_name], shuffle=True)
        return DataLoader(self.tokenized_dataset[split_name], batch_size=args['batch_size'], num_workers=args['num_workers'], sampler=sampler)

    def train_dataloader(self):
        return self._get_dataloader('train')

    def val_dataloader(self):
        return self._get_dataloader('valid')

    def test_dataloader(self):
        return self._get_dataloader('test')


genome_llama2 = GenomeLLAMA2(tokenizer, tokenizedDataset, args['lr'])


# Initialize with a seed
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# Auto Wrap Policy
policy = partial(size_based_auto_wrap_policy, min_num_params=1000000)

# ModelCheckpoint Callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',  # Replace 'val_loss' with your validation metric
    dirpath='genome_llama2_checkpoint',  # Path where the checkpoints will be saved
    filename='genome_llama2-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,  # Number of best models to save; set to -1 to save all checkpoints
    mode='min',  # 'min' for metrics where lower is better (like loss), 'max' for accuracy
)

# EarlyStopping Callback
early_stop_callback = EarlyStopping(
    monitor='val_loss',  # or another metric
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    verbose=True,
    mode='min'
)

# Logger
logger = TensorBoardLogger("genome_llama2_tb_logs", name="genome_llama2")

# FSDP Strategy
strategy = FSDPStrategy(
    sharding_strategy='FULL_SHARD',
    auto_wrap_policy=policy,
    limit_all_gathers=True,
    cpu_offload=True,
    mixed_precision='mixed',  # Ensure mixed precision is enabled
    )

# Define lightning trainer
trainer = pl.Trainer(
    accelerator="gpu",
    devices=args['gpus'],
    max_epochs=args['epochs'],
    precision=args['precision'],
    strategy=strategy,
    accumulate_grad_batches=args['accumulate_grad_batches'],
    callbacks=[checkpoint_callback, early_stop_callback],
    enable_checkpointing=args['enable_checkpointing'],
    logger=logger,
    profiler=args['profiler']
    )

print("Started training")
trainer.fit(genome_llama2)
print("Success")
