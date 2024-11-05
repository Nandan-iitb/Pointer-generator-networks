from code_2 import train, MyDataset, Vocab, Dataset, simple_tokenizer, Parameters
import torch

params =Parameters()
all_in_one_dataset = Dataset(params.all_in_one_path, simple_tokenizer, params.max_src_len, params.max_tgt_len, max_rows=1e5,
                        truncate_src=True, truncate_tgt=True)
# convert the embeddings to a tensor

vocab = all_in_one_dataset.build_vocab(params.vocab_min_frequency, embed_file=params.embed_file)
embedding_weights = torch.from_numpy(vocab.embeddings)
test_data=MyDataset([pair[0] for pair in all_in_one_dataset.pairs[-7226:]],[pair[1] for pair in all_in_one_dataset.pairs[-7226:]],vocab)
outputs = []

train(all_in_one_dataset,vocab, params, embedding_weights,learning_rate=0.001,num_epochs = 25)