import torch

def extract_task_emb(model):
    return model.prefix_encoder.weight.mean(0)

def extract_ptuning_v2_task_emb(model):
    return model.prefix_encoder.embedding.weight.mean(0)

def extract_lora_task_emb(model):
    para_list = []
    for name, param in self.bert.named_parameters():
        if 'lora' in name:
            para_list.append(param.reshape(-1))
    return torch.cat(para_list)

def extract_bitfit_task_emb(model):
    para_list = []
    for name, param in self.bert.named_parameters():
        if 'bias' in name:
            para_list.append(param.reshape(-1))
    return torch.cat(para_list)



# if __name__ == '__main__':
#     from transformers import AutoTokenizer, RobertaConfig
#     from model.sequence_classification import RobertaPromptForSequenceClassification, RobertaPrefixForSequenceClassification
#     config = RobertaConfig.from_pretrained('roberta-base')
#     config.pre_seq_len = 2
#     model = RobertaPromptForSequenceClassification.from_pretrained('roberta-base', config=config)
#     print(extract_task_emb(model).shape)

#     config.prefix_projection = False
#     model = RobertaPrefixForSequenceClassification.from_pretrained('roberta-base', config=config)

#     print(extract_ptuning_v2_task_emb(model).shape)

