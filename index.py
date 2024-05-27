from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset
from torch.utils.data import DataLoader
import streamlit as st

# Load and preprocess dataset
df = pd.read_csv('hospital_support.csv', sep=';', skiprows=1)

def preprocess(text):
    return text.lower()

df['question'] = df['question'].apply(preprocess)
df['answer'] = df['answer'].apply(preprocess)

# Prepare texts for training the tokenizer
texts = df['question'].tolist() + df['answer'].tolist()

# Train custom tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=[
                              "<s>", "</s>", "<unk>", "<pad>", "<mask>"])
tokenizer.train_from_iterator(texts, trainer)
tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> <s> $B </s>",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)
tokenizer.enable_truncation(max_length=128)  # Reduced max_length for memory efficiency
tokenizer.enable_padding(
    pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")

# Save and load tokenizer
tokenizer.save("custom_tokenizer.json")
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

# Define a simple sequence-to-sequence model using LSTM
class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0.5):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim,
                               n_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim,
                               n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg):
        embedded_src = self.embedding(src)
        embedded_trg = self.embedding(trg)
        _, (hidden, cell) = self.encoder(embedded_src)
        outputs, _ = self.decoder(embedded_trg, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions

# Define the model parameters
input_dim = tokenizer.get_vocab_size()
embedding_dim = 256  # Dimension of the embeddings
hidden_dim = 256
output_dim = tokenizer.get_vocab_size()
n_layers = 2
model = Seq2SeqModel(input_dim, embedding_dim, hidden_dim, output_dim, n_layers)

# Tokenization function
def tokenize_function(examples):
    inputs = [ex for ex in examples['question']]
    targets = [ex for ex in examples['answer']]
    model_inputs = tokenizer.encode_batch(inputs)
    labels = tokenizer.encode_batch(targets)
    model_inputs = {'input_ids': [x.ids for x in model_inputs]}
    labels = {'labels': [x.ids for x in labels]}
    return {**model_inputs, **labels}

# Convert dataframe to Dataset
dataset = Dataset.from_pandas(df)
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["question", "answer"])

# Split dataset into training and testing sets
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# DataLoader for batching
def collate_fn(batch):
    input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
    labels = [torch.tensor(item['labels'], dtype=torch.long) for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.token_to_id('<pad>'))
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=tokenizer.token_to_id('<pad>'))
    return {'input_ids': input_ids, 'labels': labels}

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)

# Training loop
def train(model, train_dataloader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            src = batch['input_ids']
            trg = batch['labels']
            output = model(src, trg)
            loss = criterion(output.view(-1, output_dim), trg.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_dataloader)}')

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id('<pad>'))

# Train the model
train(model, train_dataloader, optimizer, criterion, epochs=5)  # Increased epochs

# Function to generate chatbot responses
def chatbot_response(question):
    inputs = tokenizer.encode(question).ids
    inputs = torch.tensor(inputs, dtype=torch.long).unsqueeze(0)
    trg = torch.zeros((1, 128), dtype=torch.long).fill_(tokenizer.token_to_id('<pad>'))
    with torch.no_grad():
        output = model(inputs, trg)
    output_ids = output.argmax(-1).squeeze().tolist()
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    return response

# Streamlit interface
def main():
    st.title("Hospital Support Chatbot")
    st.write("Welcome to the Hospital Support Chatbot. Please type a message and press Enter to start the conversation.")

    user_input = st.text_input("You:")
    if user_input:
        response = chatbot_response(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None)
        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have a great day!")
            st.stop()

if __name__ == '__main__':
    main()
