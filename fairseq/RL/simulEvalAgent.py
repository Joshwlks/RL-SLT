import torch
from fairseq.RL.interactive import Generator
import sentencepiece as spm
import fileinput

from simuleval.agents import TextAgent
from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
from simuleval.states import TextStates


class BaseTextAgent(TextAgent):
    data_type="text"

    def __init__(self, args):
        super().__init__(args)
        # init your agent here, i.e. load model and vocab etc
        # create SentencePiece encoder and decoder
        self.m=spm.SentencePieceProcessor(model_file='/home/bhaddow/experiments/e2e-slt-noise/data/baseline-mt/en-fr/spm_unigram8000_en.model')
        self.m_out=spm.SentencePieceProcessor(model_file='/home/bhaddow/experiments/e2e-slt-noise/data/baseline-mt/en-fr/spm_unigram8000_fr.model')
        # load model
        self.model=Generator("/home/bhaddow/experiments/e2e-slt-noise/data/baseline-mt/en-fr/data-bin", "/home/bhaddow/experiments/e2e-slt-noise/expts/baseline-mt/en-fr/checkpoints/checkpoint_best.pt")
        # set dictionary
        self.dict={}
        self.dict['tgt']=self.model.tgt_dict
        # prep model for generating
        self.model=self.model.generate
        self.prev_tokens=None
        # Keep writing until the full source word is completed
        self.sp_len=0
        # Read Write agent counter
        self.i=0

    def build_states(self, args, client, sentence_id):
        states = TextStates(args, client, sentence_id, self)
        self.initialize_states(states)
        #self.prev_tokens=None
        return states

    def initialize_states(self, states):
        states.stream = dict()
        states.stream["STREAM"] = ""

    def preprocess(self, states):
        # Get the current read words and convert to sp
        STREAM=" ".join(states.segments.source.value)  
        if '</s>' in STREAM:
            STREAM=STREAM.replace('</s>', '')      
        STREAM=self.m.encode(STREAM, out_type=str)
        # Get the difference between the prev and new sp lens
        self.sp_len=len(STREAM) - len(states.stream["STREAM"].split(' ')) if states.stream["STREAM"] != '' else len(STREAM)
        # Set the stream to be the current sp encoded source
        STREAM=" ".join(STREAM)
        states.stream["STREAM"] = STREAM
        print(f"the stream: {STREAM}")
        return states
    
    def postprocess(self, states):
        _, new_prev_tokens, lprobs=self.model(states.stream["STREAM"], self.prev_tokens)
        index=torch.argmax(lprobs).item()
        tgt_word=self.dict['tgt'][index] 

        if tgt_word[0]!='▁' and tgt_word!='</s>':
            self.prev_tokens=new_prev_tokens
            return tgt_word
        else:
            return None

    # def policy(self, states):
    #     # Make random decision to read or write here
    #     print(f"the source read so far: {states.segments.source.value}")
    #     return READ_ACTION if torch.rand(1).item()>0.5 and not states.finish_read() or states.segments.source.length()==0 else WRITE_ACTION

    # def policy(self, states):
    #     # Read Write policy
    #     print(f"the source read so far: {states.segments.source.value}")
    #     self.i+=1
    #     return READ_ACTION if self.i%2==1 and not states.finish_read() or states.segments.source.length()==0 else WRITE_ACTION

    def policy(self, states):
        # Wait until end policy
        return READ_ACTION if not states.finish_read() else WRITE_ACTION
                    
    def predict(self, states):
        # predict the token here
        states=self.preprocess(states)
        tgt_word=[]
        # Handle Sentence Piecing on input
        for i in range(max(self.sp_len,1)):
            _, self.prev_tokens, lprobs=self.model(states.stream["STREAM"], self.prev_tokens)
            index=torch.argmax(lprobs).item()
            tgt_word.append(self.dict['tgt'][index])
            
        if self.dict['tgt'][2] in tgt_word:
            self.prev_tokens=None
            print(f"the tgt words: {states.segments.target.value}")
            return self.dict['tgt'][2]                  
        else:
            # Handle Sentence Piecing on output
            rest_of_word=self.postprocess(states)
            while rest_of_word is not None:
                tgt_word.append(rest_of_word)
                rest_of_word=self.postprocess(states)
            return self.m_out.decode(tgt_word)