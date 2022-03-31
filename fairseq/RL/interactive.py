"""
Version of the fairseq-interactive feature that can be used within scripts without the need to call to a subprocess :)

"""
import fileinput
from collections import namedtuple
import os
import sys
import math

import numpy as np

import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import data_utils

import warnings
warnings.filterwarnings('ignore', '.*floor_divide is deprecated.*',)

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')

def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], 
            src_lengths=batch['net_input']['src_lengths'],
        )

def buffered_read(input, buffer_size):
        buffer = []
        #here the input for interactive is read, the fileinput is a python class. The input method takes a file or if passed '-' will read from sys.stin
        with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h: 
            for src_str in h:
                buffer.append(src_str.strip())
                if len(buffer) >= buffer_size:
                    yield buffer
                    buffer = []

        if len(buffer) > 0:
            yield buffer


class Generator():
    def __init__(self, data_path, checkpoint_path="checkpoint_best.pt"):
        self.parser = options.get_generation_parser(interactive=True)
        self.parser.set_defaults(path=checkpoint_path,
            remove_bpe="sentencepiece", dataset_impl="lazy", num_wokers=5
        )
        self.args = options.parse_args_and_arch(self.parser, 
            input_args=[data_path]
        )

        utils.import_user_module(self.args)

        if self.args.buffer_size < 1:
            self.args.buffer_size = 1
        if self.args.max_tokens is None and self.args.batch_size is None:
            self.args.batch_size = 1

        assert not self.args.sampling or self.args.nbest == self.args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not self.args.batch_size or self.args.batch_size <= self.args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        self.use_cuda = torch.cuda.is_available() and not self.args.cpu

        self.task = tasks.setup_task(self.args)

        self.models, self._model_args = checkpoint_utils.load_model_ensemble(
            self.args.path.split(':'),
            arg_overrides=eval(self.args.model_overrides),
            task=self.task,
        )
   
        # making the dictionary objects
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=None if self.args.no_beamable_mm else self.args.beam,
                need_attn=self.args.print_alignment,
            )
            if self.args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        self.generator = self.task.build_generator(self.models, self.args)

        if self.args.remove_bpe == 'gpt2':
            from fairseq.gpt2_bpe.gpt2_encoding import get_encoder
            self.decoder = get_encoder(
                'fairseq/gpt2_bpe/encoder.json',
                'fairseq/gpt2_bpe/vocab.bpe',
            )
            self.encode_fn = lambda x: ' '.join(map(str, self.decoder.encode(x)))
        else:
            self.decoder = None
            self.encode_fn = lambda x: x

        self.align_dict = utils.load_align_dict(self.args.replace_unk)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(),
            *[model.max_positions() for model in self.models]
        )

    # for an input or for a file... I am just using the input feature atm.
    def generate(self, input, previous_tokens=None ,string_input=True, rl=True):
        start_id = 0
        if string_input:
            inputs = [input] 
            results = []
            for batch in make_batches(inputs, self.args, self.task, self.max_positions, self.encode_fn):
                src_tokens = batch.src_tokens  
                src_lengths = batch.src_lengths
                if self.use_cuda:
                    src_tokens = src_tokens.cuda()
                    src_lengths = src_lengths.cuda()

                sample = {
                    'net_input': {
                        'src_tokens': src_tokens,
                        'src_lengths': src_lengths,
                    },
                }
                # Here the translations take place
                # I will also want to pass the previous output through
                translations, prev_tokens, lprobs, observation = self.task.inference_step(self.generator, self.models, sample, previous_tokens=previous_tokens)
                if rl:
                    return translations, prev_tokens, lprobs, observation
                    
                for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                    src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                    results.append((start_id + id, src_tokens_i, hypos))
                
            for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
                    if self.src_dict is not None:
                        src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
            
            if hypos==[]:
                #print(f"the prev tokens indexed: {(prev_tokens==1).nonzero(as_tuple=True)[1][0]}")
                _, hypo_str, _ = utils.post_process_prediction(
                    hypo_tokens=prev_tokens[:,:(prev_tokens==1).nonzero(as_tuple=True)[1][0]].int().cpu(),
                    src_str=src_str,
                    alignment=None, 
                    align_dict=self.align_dict, 
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.args.remove_bpe,
                )
                return hypo_str, prev_tokens, lprobs, observation

            for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                if hypo==None:
                    return None, prev_tokens, lprobs, observation

                # print(f"src_str: {src_str}")
                # print(f"alignment: {hypo['alignment'].int().cpu()}")
                # raise NotImplementedError()
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=self.align_dict,
                    tgt_dict=self.tgt_dict,
                    remove_bpe=self.args.remove_bpe,
                )
                
                if self.decoder is not None:
                    hypo_str = self.decoder.decode(map(int, hypo_str.strip().split()))

                return hypo_str, prev_tokens, attn, decoder_state

        ## File input for the NMT... not going to need to use this I do not think
        else:
            for inputs in buffered_read(input, 0):
                results = []
                for batch in make_batches(inputs, self.args, self.task, self.max_positions, self.encode_fn):
                    src_tokens = batch.src_tokens
                    src_lengths = batch.src_lengths
                    if self.use_cuda:
                        src_tokens = src_tokens.cuda()
                        src_lengths = src_lengths.cuda()

                    sample = {
                        "net_input": {
                            "src_tokens": src_tokens,
                            "src_lengths": src_lengths,
                        },
                    }
                    translations = self.task.inference_step(self.generator, self.models, sample)
                    for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                        src_tokens_i = utils.strip_pad(src_tokens[i], self.tgt_dict.pad())
                        results.append((start_id + id, src_tokens_i, hypos))
                
                for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
                    if self.src_dict is not None:
                        src_str = self.src_dict.string(src_tokens, self.args.remove_bpe)
                
                for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=self.align_dict,
                        tgt_dict=self.tgt_dict,
                        remove_bpe=self.args.remove_bpe,
                    )
                    
                    if self.decoder is not None:
                        hypo_str = self.decoder.decode(map(int, hypo_str.strip().split()))

                    return hypo_str
                
                # update running id_ counter
                start_id += len(inputs)
                       
        

if __name__ == '__main__':
    gen = Generator("/home/bhaddow/experiments/e2e-slt-noise/data/baseline-mt/en-fr/data-bin", "/home/bhaddow/experiments/e2e-slt-noise/expts/baseline-mt/en-fr/checkpoints/checkpoint_best.pt")
