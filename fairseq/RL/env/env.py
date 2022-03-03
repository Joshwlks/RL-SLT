import math
from typing import Optional

import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy as np

import torch
from fairseq.RL.interactive import Generator
import sentencepiece as spm
import fileinput

from fairseq.RL.env.reward.bleu import sentence_bleu, SmoothingFunction
from fairseq.RL.env.reward.latency import averageProportion, consecutiveWaitDelay

### Input from file ###
def get_sentences(input):
        buffer = []
        #here the input for interactive is read, the fileinput is a python class. The input method takes a file or if passed '-' will read from sys.stin
        with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h: 
            for _str in h:
                buffer.append(_str.strip())
        return buffer


class SNmtEnv(gym.Env):
    """
    Description: Simultaneous neural machine translation environment. Contains the NMT model from fairseq and the source and target sentences. Recieves READ or WRITE action and updates the state of 
    the environment via the NMT model. Outputs the new observation and reward given the agent action.
    Source: Fairseq
    Observations:
        Type: Box(N,) where N is the length of the source sentence + decoder embedding size + 1
        Description: concatenation of cross attention vector, decoder embedding and the predicted target token 
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Read next word from source sentence
        1     Write word to target sentence

    Rewards: sum of the change in translation quality (smoothed BLEU) and the latency (AP, CW).
    Starting State:
    Episode Termination:
    """
    def __init__(self, source_input='/home/jwilkins/RL-SLT/fairseq/fairseq/RL/source.txt', target_input='/home/jwilkins/RL-SLT/fairseq/fairseq/RL/target.txt', **kwargs):
        # load model and decoder embedding ## Note that these may error if you change the model as in fairseq there are different conditions
        self.model=Generator("/home/bhaddow/experiments/e2e-slt-noise/data/baseline-mt/en-fr/data-bin", "/home/bhaddow/experiments/e2e-slt-noise/expts/baseline-mt/en-fr/checkpoints/checkpoint_best.pt")
        self.obs_dim=self.model.models[0].decoder.embed_dim * 4
        self.embed_tokens=self.model.models[0].decoder.embed_tokens
        self.embed_scale=self.model.models[0].decoder.embed_scale
        self.embed_positions=self.model.models[0].decoder.embed_positions

        ## define spaces
        self.action_space=spaces.Discrete(2)
        self.observation_space=spaces.Discrete(2)
        #self.observation_space=spaces.Box(np.array([-np.inf]*self.obs_dim),np.array([np.inf]*self.obs_dim), dtype=np.float64)

        # sentence piece encodeing and decoding models
        self.sp_encoder=spm.SentencePieceProcessor(model_file='/home/bhaddow/experiments/e2e-slt-noise/data/baseline-mt/en-fr/spm_unigram8000_en.model').encode
        self.sp_decoder=spm.SentencePieceProcessor(model_file='/home/bhaddow/experiments/e2e-slt-noise/data/baseline-mt/en-fr/spm_unigram8000_fr.model').decode

        # set dictionary
        self.dict={}
        self.dict['tgt']=self.model.tgt_dict

        # prep model for generating
        self.model=self.model.generate
        self.prev_tokens_predicted=None
        self.prev_tokens_actual=None

        # done
        self.done=False

        # src sentences
        self.src_sents=get_sentences(source_input)
        self.sent_num=0

        # ref translation sentence
        self.refs=get_sentences(target_input)
        self.ref=[self.refs[self.sent_num].split()]

        # current src stream
        self.curr_src=self.src_sents[self.sent_num].split()
        self.sp_sent=self.sp_encoder(self.curr_src, out_type=str)
        self.curr_word=0
        self.stream=self.curr_src[:self.curr_word+1]

        # current target sentence
        self.curr_tgt=[]
        self.pred_tgt=[]
        self.curr_tgt_word=0

        # Finshed reading when source is fully read
        self.finished_reading=False

        # reward
        self.prev_bleu=0
        self.ap=0
        self.d=[]
        self.cw=0
        self.alpha=-0.5
        self.beta=-0.5
        self.d_star=0.8
        self.c_star=3
        self.rewards=[]
        self.latencys=[]
        self.bleus=[]
    
    def postprocess(self, to_translate):
        _, new_prev_tokens, lprobs, observation=self.model(to_translate, self.prev_tokens_predicted)
        index=torch.argmax(lprobs).item()
        tgt_word=self.dict['tgt'][index] 

        if tgt_word[0]!='▁' and tgt_word!='</s>':
            self.prev_tokens_predicted=new_prev_tokens
            return tgt_word, observation
        else:
            return None, None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        acts=["Read", "Write"]
        print(f"Action taken: {acts[action]}")
        
        #### READ or WRITE ####
        self.done=True if self.dict['tgt'][2] in self.pred_tgt and action==1 else False
        if action==0: # don't read if you have already read the sentence
            # Read next source word
            self.curr_word+=1 if not self.finished_reading else 0
            self.stream=self.curr_src[:self.curr_word+1]
            self.finished_reading=True if self.stream==self.curr_src else False
            self.ap+=1 # we only add one to ap if we actually read, cw handles the delay caused by reading when you have already fully read a sentence.
        elif action==1 and not self.done:
            # Write to target sentence
            self.curr_tgt+=self.pred_tgt
            self.curr_tgt_word+=1
            self.prev_tokens_actual=self.prev_tokens_predicted
            self.d.append(self.ap)
        assert self.done == (self.dict['tgt'][2] in self.pred_tgt) * action
        if self.done: print(f"Done but was there an end of sentence?: {self.dict['tgt'][2] in self.pred_tgt}")
        #### Calculate rewards ###
        # change in bleu
        current_bleu=sentence_bleu(self.ref,self.sp_decoder(self.curr_tgt).split() , smoothing_function=SmoothingFunction().method5)[self.done]
        change_in_bleu=current_bleu - self.prev_bleu
        self.prev_bleu=current_bleu
        self.bleus.append(change_in_bleu)

        # latency
        if self.done and self.curr_tgt_word==0:
            print("Why is it not resetting ????")
            raise NotImplementedError()
        self.cw+=1 if action==0 else -self.cw
        d_t=0 if not self.done else sum(self.d)/((self.curr_word+1)*self.curr_tgt_word)
        d_t=d_t - self.d_star if d_t>self.d_star else 0    
        latency_reward=self.alpha*(np.sign(self.cw - self.c_star) + 1) + self.beta*(d_t)
        self.latencys.append(latency_reward)
        # combine latency and quality
        reward=change_in_bleu + latency_reward
        self.rewards.append(reward)
        #### Get new observation ####
        self.prev_tokens_predicted=self.prev_tokens_actual
        self.pred_tgt=[]
        pred_token_embedding=None
        to_translate=' '.join([subitem for sublist in self.sp_sent[:self.curr_word+1] for subitem in sublist])
        
        for sp in self.sp_sent[self.curr_word]:
            _, self.prev_tokens_predicted, lprobs, observation=self.model(to_translate, self.prev_tokens_predicted)
            index=torch.argmax(lprobs).item()
            self.pred_tgt.append(self.dict['tgt'][index])

        # handle Sentence Piecing on output
        rest_of_word, temp_obs=self.postprocess(to_translate)
        while rest_of_word is not None:
            self.pred_tgt.append(rest_of_word)
            observation=temp_obs
            rest_of_word, temp_obs=self.postprocess(to_translate) 
        
        # embed predicted target token
        token_indexs=(self.prev_tokens_predicted!=1).nonzero(as_tuple=True)[1]
        tokens=torch.index_select(self.prev_tokens_predicted,1,token_indexs)
        token_postion=self.embed_positions(tokens, incremental_state=None)[:,-1:]
        token_embed=self.embed_scale * self.embed_tokens(tokens[:,-1:])
        token_embed+=token_postion

        # transform token embedding into 1D numpy array to match the rest of the observation
        token_embed=token_embed.transpose(0,1)
        token_embed=token_embed[:,-1:,:].squeeze().cpu().detach().numpy()

        # form next observation via concatenation of attn, decoder state and token embedding
        #observation=np.concatenate((observation, token_embed), axis=0)
        observation=np.array([self.curr_word+1,self.curr_tgt_word])

        if self.done: print(f"the total reward for this episode was: {sum(self.rewards)}")
        return observation, reward, self.done, {}

    def reset(self):
        # set up for the next sentence
        self.sent_num+=1 if self.done else 0
        self.done=False
        self.rewards=[]
        self.bleus=[]
        self.latencys=[]

        # current src stream
        self.curr_src=self.src_sents[self.sent_num].split()
        self.sp_sent=self.sp_encoder(self.curr_src, out_type=str)
        self.curr_word=0
        self.stream=self.curr_src[:self.curr_word+1]
        self.prev_tokens_actual=None
        self.prev_tokens_predicted=None
                                         
        # reset tgt
        self.curr_tgt=[]
        self.pred_tgt=[]
        self.curr_tgt_word=0

        # reset reward
        self.prev_bleu=0
        self.ap=0
        self.cw=0

        # get first observation by translating first word in sentence
        to_translate=' '.join([subitem for sublist in self.sp_sent[:self.curr_word+1] for subitem in sublist])
        for sp in self.sp_sent[self.curr_word]:
                _, self.prev_tokens_predicted, lprobs, observation=self.model(to_translate, self.prev_tokens_predicted)
                index=torch.argmax(lprobs).item()
                self.pred_tgt.append(self.dict['tgt'][index])
        # Handle Sentence Piecing on output
        rest_of_word, temp_obs=self.postprocess(to_translate)
        while rest_of_word is not None:
            self.pred_tgt.append(rest_of_word)
            observation=temp_obs
            rest_of_word, temp_obs=self.postprocess(to_translate)

        # embed predicted target token
        token_indexs=(self.prev_tokens_predicted!=1).nonzero(as_tuple=True)[1]
        tokens=torch.index_select(self.prev_tokens_predicted,1,token_indexs)
        token_postion=self.embed_positions(tokens, incremental_state=None)[:,-1:]
        token_embed=self.embed_scale * self.embed_tokens(tokens[:,-1:])
        token_embed+=token_postion

        # transform token embedding into 1D numpy array to match the rest of the observation
        token_embed=token_embed.transpose(0,1)
        token_embed=token_embed[:,-1:,:].squeeze().cpu().detach().numpy()

        # form next observation via concatenation of attn, decoder state and token embedding
        #observation=np.concatenate((observation, token_embed), axis=0)
        observation=np.array([self.curr_word+1,self.curr_tgt_word])
        return observation


if __name__=="__main__":
    from fairseq.RL.train_model import make_vec_envs
    env=make_vec_envs("EnFrSNmtEnv-v0", 50, 1)
    obs=env.reset()
    
    env.step([0])
    env.step([0])
    env.step([1])
    env.step([1])
    env.step([1])
    env.step([1])
    env.step([1])
    env.step([0])
    env.step([1])
    env.step([1])
    env.step([1])
    env.step([0])
    env.step([0])
    env.step([0])
    env.step([1])
    env.step([0])
    env.step([1])

    print("episode 2 should begin here")
    env.step([0])
    env.step([0])
    env.step([1])
    env.step([1])
    env.step([1])
    env.step([1])
    env.step([1])
    env.step([0])
    env.step([1])
    env.step([1])
    env.step([1])
    env.step([0])
    env.step([0])
    env.step([0])
    env.step([1])
    env.step([0])
    env.step([1])      

