#Valiation Script n-class
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertForSequenceClassification, DistilBertModel, AutoTokenizer
from src.models.ml_utils import create_bert_tokens, create_attention_mask
import numpy as np 
import os 

class DistilBERTClass(torch.nn.Module):
    def __init__(self, num_class, saved = False, model_path = ''):
        super(DistilBERTClass, self).__init__()
        if saved:
            self.l1 = DistilBertModel.from_pretrained(model_path, local_files_only = True)
        else:
            self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_class)
    def forward(self, input_ids, input_mask):
        output_1 = self.l1(input_ids, attention_mask = input_mask)
        hidden_state = output_1[0]
        x = hidden_state[:,0]
        x = self.pre_classifier(x)
        x = torch.nn.ReLU()(x)
        x = self.dropout(x)
        output = self.classifier(x)
        return output

def prediction_data_creation(bert_tokens, attention_masks, batch_size = 32): 
    test_inputs = torch.tensor(bert_tokens)
    test_masks = torch.tensor(attention_masks)
    prediction_data = TensorDataset(test_inputs, test_masks)
    prediction_dataloader = DataLoader(prediction_data, batch_size=batch_size)
    return prediction_dataloader

def Evaluate_model(model, prediction_dataloader, num_class, device):
    model.eval()
    predictions = np.empty([0, num_class]) 
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            outputs = model(b_input_ids, input_mask=b_input_mask)
      #logits = outputs[0]
      #logits = logits.T[0].detach().cpu().numpy()
        predictions = np.vstack((predictions, outputs.detach().cpu().numpy()))
    return predictions

def Evaluate_model_twoClass(model, prediction_dataloader, device):
    model.eval()
    predictions = [] 
    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            outputs = model(b_input_ids,attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.T[0].detach().cpu().numpy()
        for l in logits:
            predictions.append(l)
    return predictions

def test_on_pretrained_model(df, text_col, model_location, base_location, use_saved, num_class, device, batch_size = 32, MAX_LEN = 100):
    if use_saved:
        tokenizer = AutoTokenizer.from_pretrained(base_location, do_lower_case=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    input_ids = create_bert_tokens(tokenizer, df, text_col, MAX_LEN)
    attention_masks = create_attention_mask(input_ids)
    prediction_dataloader = prediction_data_creation(input_ids, attention_masks, batch_size) 
    if num_class > 2:
        model_path = os.path.join(os.getcwd(), model_location, 'trained_model.bin')
        model = DistilBERTClass(num_class, use_saved, base_location)
        model.load_state_dict(torch.load(model_path))
    else:
        model_path = os.path.join(os.getcwd(), model_location, 'pytorch_model.bin')
        model_cnf = os.path.join(os.getcwd(), model_location, 'config.json')
        model = DistilBertForSequenceClassification.from_pretrained(model_path, config = model_cnf, local_files_only = True)
    #print(model_path)
    if device == 'cuda': 
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('Current GPU in use:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, although GPU option was selected. Using the CPU instead.')
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print('Using CPU to run the model. You can expect an extended runtime. Use GPU instead.')
    model.to(device)
    if num_class > 2:
        predictions = Evaluate_model(model, prediction_dataloader, num_class, device)
    else:
        predictions = Evaluate_model_twoClass(model, prediction_dataloader, device)
    return predictions  


    
