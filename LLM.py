from transformers import AutoModel, AutoTokenizer
from torch import nn
import torch

class LLM(nn.Module):
    def __init__(self, dim, llm_ckp_dir, device, dropout=0.1):
        super(LLM, self).__init__()

        '''
        in our model, we use the frozen llama 3.2 1B for lightweight LLM, 
        taking into account both semantic processing and computing efficiency/performance
        '''
        self.llama = AutoModel.from_pretrained(
            llm_ckp_dir,
            device_map=device,
        )

        self.llama_tokenizer = AutoTokenizer.from_pretrained(llm_ckp_dir)

        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        # freeze the parameters of LLM, don't train
        for name, param in self.llama.named_parameters():
            param.requires_grad = False

        # multimodal fusion network FFN
        self.fusion = nn.Sequential(
            nn.Linear(2*dim, 2*dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*dim, self.llama.config.hidden_size),
            ).to(device)


    def forward(self, x_time, x_news):
        x = torch.cat([x_news, x_time], dim=-1) # concatenate the two modality: numerical market feature and news
        x = self.fusion(x) # multimodal fusion network
        return self.llama(inputs_embeds=x).last_hidden_state # process the concatenated feature through LLM and obtain the last_hidden_state