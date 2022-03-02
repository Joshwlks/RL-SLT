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
    #gen = Generator("/home/jwilkins/RL-SLT/fairseq/examples/wmt19/en-de/data-bin", "/home/jwilkins/RL-SLT/fairseq/examples/wmt19/en-de/wmt19.en-de.ffn8192.pt")

    #print(gen.generate(input='fairseq/RL/source.txt'))  # was slightly wrong to get the NMT to work on a document basis I think...
    #tokens = gen.generate("▁Hello ▁my ▁friends ▁good ▁morning .")[1]
    lprob1, lprob2 = (0,0)

    sent1, tokens, lprob1, observation = gen.generate("▁Please ▁all ▁make")
    
    #print(f"the sent: {sent1}")
    # print(f"the attn tensor: {attn.size()}")
    # print(f"the decoder state: {decoder_state}")
    # #print(f"the max lprob is {torch.max(lprob1)} at index {torch.argmax(lprob1)}")
    # #print(f"the tokens: {tokens}")
    sent2, tokens, lprob2, observation = gen.generate("▁Please ▁all ▁make ▁yourselves ▁comfortable", previous_tokens=tokens)
    # print(f"the sent: {sent2}")
    sent3, tokens, lprob2, observation = gen.generate("▁Please ▁all ▁make ▁yourselves ▁comfortable", previous_tokens=tokens)
    # print(f"the sent: {sent3}")
    sent4, tokens, lprob2, observation = gen.generate("▁Please ▁all ▁make ▁yourselves ▁comfortable", previous_tokens=tokens)
    # print(f"the sent: {sent4}")
    # print(f"the attn tensor: {attn.size()}")
    # print(f"the decoder state: {decoder_state}")
    index=torch.argmax(lprob2).item()
    print(f"the index: {index}")
    #pred_tgt.append(self.dict['tgt'][index])
    print(attn.shape)
    print(decoder_state.shape)
    print(obs.shape)
    print(obs)
    
    


    # #print(f"the max lprob is {torch.max(lprob2)} at index {torch.argmax(lprob2)}")
    # #print(f"the tokens: {tokens}")
    # sent3, tokens, lprob3 = gen.generate("▁Please ▁all ▁make, ▁yourselves ▁comfortable", previous_tokens=tokens)
    # #print(f"the max lprob is {torch.max(lprob3)} at index {torch.argmax(lprob3)}")
    # #print(f"the tokens: {tokens}")
    # sent4, tokens, lprob4 = gen.generate("▁Please ▁all ▁make, ▁yourselves ▁comfortable", previous_tokens=tokens)
    # #print(f"the max lprob is {torch.max(lprob4)} at index {torch.argmax(lprob4)}")
    # #print(f"the tokens: {tokens}")
    # sent5, tokens, lprob5 = gen.generate("▁Please ▁all ▁make, ▁yourselves ▁comfortable", previous_tokens=tokens)
    # #print(f"the max lprob is {torch.max(lprob5)} at index {torch.argmax(lprob5)}")
    # #print(f"the tokens: {tokens}")
    # sent6, tokens, lprob5 = gen.generate("▁Please ▁all ▁make, ▁yourselves ▁comfortable", previous_tokens=tokens)
    # #print(f"the max lprob is {torch.max(lprob5)} at index {torch.argmax(lprob5)}")
    # sent7, tokens, lprob5 = gen.generate("▁Please ▁all ▁make, ▁yourselves ▁comfortable", previous_tokens=tokens)
    # sent8, tokens, lprob5 = gen.generate("▁Please ▁all ▁make, ▁yourselves ▁comfortable", previous_tokens=tokens)
    # sent9, tokens, lprob5 = gen.generate("▁Please ▁all ▁make, ▁yourselves ▁comfortable", previous_tokens=tokens)
    # print(sent1 + "\n", sent2 + '\n', sent3 + '\n', sent4 + '\n', sent5 + '\n', sent6 + '\n' + sent7 + '\n' + sent8 + '\n' + sent9)
    # print(sent1.split(" ")[-1])
    # print(sent2.split(" ")[-1])
    # print(sent3.split(" ")[-1])
    # print(sent4.split(" ")[-1])
    # print(sent5.split(" ")[-1])
    # print(sent6.split(" ")[-1])


    #sent_ful, tokens, lprob5 = gen.generate("▁Please ▁all ▁make, ▁yourselves ▁comfortable")














    """

    Namespace(all_gather_list_size=16384, amp=False, amp_batch_retries=2, amp_init_scale=128, amp_scale_window=None, azureml_logging=False, batch_size=None, batch_size_valid=None, beam=1, best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, buffer_size=0, checkpoint_shard_count=1, checkpoint_suffix='', combine_valid_subsets=None, constraints=None, cpu=False, cpu_offload=False, criterion='cross_entropy', curriculum=0, data='/home/jwilkins/RL-SLT/fairseq/examples/wmt19/en-de/data-bin', data_buffer_size=10, dataset_impl='lazy', ddp_backend='pytorch_ddp', ddp_comm_hook='none', decoding_format=None, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_num_procs=4, distributed_port=-1, distributed_rank=0, distributed_world_size=1, diverse_beam_groups=-1, diverse_beam_strength=0.5, diversity_rate=-1.0, empty_cache_freq=0, eos=2, eval_bleu=False, eval_bleu_args='{}', eval_bleu_detok='space', eval_bleu_detok_args='{}', eval_bleu_print_samples=False, eval_bleu_remove_bpe=None, eval_tokenized_bleu=False, fast_stat_sync=False, find_unused_parameters=False, finetune_from_model=None, fix_batches_to_gpus=False, fixed_validation_seed=None, force_anneal=None, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, fp32_reduce_scatter=False, gen_subset='test', gradient_as_bucket_view=False, heartbeat_timeout=-1, ignore_unused_valid_subsets=False, input='-', iter_decode_eos_penalty=0.0, iter_decode_force_max_iter=False, iter_decode_max_iter=10, iter_decode_with_beam=1, iter_decode_with_external_reranker=False, keep_best_checkpoints=-1, keep_interval_updates=-1, keep_interval_updates_pattern=-1, keep_last_epochs=-1, left_pad_source=True, left_pad_target=False, lenpen=1, lm_path=None, lm_weight=0.0, load_alignments=False, load_checkpoint_on_all_dp_ranks=False, localsgd_frequency=3, log_file=None, log_format=None, log_interval=100, lr_scheduler='fixed', lr_shrink=0.1, match_source_len=False, max_len_a=0, max_len_b=200, max_source_positions=1024, max_target_positions=1024, max_tokens=None, max_tokens_valid=None, max_valid_steps=None, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False, memory_efficient_fp16=False, min_len=1, min_loss_scale=0.0001, model_overrides='{}', model_parallel_size=1, nbest=1, no_beamable_mm=False, no_early_stop=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=False, no_repeat_ngram_size=0, no_reshard_after_forward=False, no_save=False, no_save_optimizer_state=False, no_seed_provided=False, nprocs_per_node=4, num_batch_buckets=0, num_shards=1, num_wokers=5, num_workers=1, on_cpu_convert_precision=False, optimizer=None, optimizer_overrides='{}', pad=1, path='/home/jwilkins/RL-SLT/fairseq/examples/wmt19/en-de/wmt19.en-de.ffn8192.pt', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, plasma_path='/tmp/plasma', post_process=None, prefix_size=0, print_alignment=None, print_step=False, profile=False, quantization_config_path=None, quiet=False, remove_bpe='sentencepiece', replace_unk=None, required_batch_size_multiple=8, required_seq_len_multiple=1, reset_dataloader=False, reset_logging=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', results_path=None, retain_dropout=False, retain_dropout_modules=None, retain_iter_history=False, sacrebleu=False, sampling=False, sampling_topk=-1, sampling_topp=-1.0, save_dir='checkpoints', save_interval=1, save_interval_updates=0, score_reference=False, scoring='bleu', seed=1, shard_id=0, simul_type=None, skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', slowmo_momentum=None, source_lang=None, suppress_crashes=False, target_lang=None, task='translation', temperature=1.0, tensorboard_logdir=None, threshold_loss_scale=None, tokenizer=None, tpu=False, train_subset='train', truncate_source=False, unk=3, unkpen=0, unnormalized=False, upsample_primary=-1, use_plasma_view=False, use_sharded_state=False, user_dir=None, valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, wandb_project=None, warmup_updates=0, write_checkpoints_asynchronously=False, zero_sharding='none'

    """
    #tokens, a_lprob, prev_states2 = gen.generate("▁Hello ▁my")
    #print(gen.generate("▁Hello ▁my ▁friends"))



   

    ### With this I discovered that the incremental states are not equal because the encoder inputs are different despite the 
    ### previous tokens being the same
    # print(type(prev_states2[0][0]))
    # equal = True
    # count = -1
    # for (k1,v1),(k2,v2) in zip(prev_states1[0][0].items(), prev_states2[0][0].items()):
    #     count += 1
    #     if type(v1) == dict:
    #         v1 = [v1[k] for k in v1.keys()][0]
    #     if type(v2) == dict:
    #         v2 = [v2[k] for k in v2.keys()][0]
    #     if not torch.equal(v1, v2):
    #         equal = False
    #         print("v1: ",v1)
    #         print(f"v2: {v2}")
    #         break
    # print(f"The count at which they stopped being equal is : {count}")
    # print(f"Are the two reorder states equal? : {torch.equal(prev_states1[1], prev_states2[1])}")
    
    #lprobs = lprob1+lprob2
    # lprobs[:, 2] = -math.inf
    # lprobs[:, 6] = -math.inf
    # lprobs[:, 4] = -math.inf
    # lprobs[:, 311] = -math.inf
    # lprobs[:, 14] = -math.inf
    # lprobs[:, 17] = -math.inf
    # lprobs[:, 74] = -math.inf
    #print(f"the max lprob is {torch.max(lprob1)} at index {torch.argmax(lprob1)}")
    #print(f"the values at 106 and 190 are the following: {lprob1[:, 106]} {lprob1[:, 190]}")
    #print(f"the max actual lprob is {torch.max(a_lprob)} at index {torch.argmax(a_lprob)}")


    # print(gen.generate("▁Hello ▁my ▁friends ▁good ▁morning .", torch.tensor([[2, 4514, 190, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0'))[0])
    #print(gen.generate("▁Hello ▁my ▁friends ▁good ▁morning .")[0])

    #the hypo tokens before : tensor([4514,  190,  736, 1336, 1098,    8,    6,    2], dtype=torch.int32)
    #the hypo tokens after : tensor([7992, 1512, 7993, 7994, 7995,    2], dtype=torch.int32)
    #the hypo str: Bonjour mes amis bons matins.
    #Bonjour mes amis bons matins.


    #the lprobs: tensor([[[-12.1199,     -inf, -11.4593,  ..., -12.0818, -12.0856, -12.1300]]],device='cuda:0')
    # the scores: tensor([[[-0.0237, -0.2651]]], device='cuda:0')
    # the tokens: tensor([[   2, 4514,  190]], device='cuda:0')
    # the original batch idxs: tensor([0], device='cuda:0')
    # the candidate indices: tensor([[ 736, 1336]], device='cuda:0')
    # the active hypo after this single step: tensor([[0]], device='cuda:0')

    """
    the value of x before decoding: tensor([[[ 1.5653e+00, -1.8733e+00,  3.7289e-01,  2.6690e+00, -2.4215e+00,
          -3.7343e-01,  2.3743e+00,  9.6193e-01, -1.9239e+00,  6.7861e-01,
           6.1382e-01, -2.4516e-02,  3.1538e+00, -3.8325e-01,  1.2373e+00,
           7.8511e-01, -1.2753e-01,  2.7837e-01, -2.3844e+00,  1.4114e+00,
           8.1479e-01,  1.8077e-02,  2.4631e-01, -1.9357e+00,  1.1097e+00,
           1.4653e+00,  2.1345e-01, -1.9639e+00,  1.2375e+00, -2.9968e-01,
           2.2151e-01,  2.1866e+00,  1.6082e+00,  1.4597e-02,  2.0265e-01,
           3.1167e+00, -1.7203e+00, -1.6692e+00, -6.6640e-02, -2.4517e+00,
           8.4164e-01, -4.1226e-02,  3.4967e+00,  6.0434e-01,  3.3177e+00,
           1.0782e+00,  6.4751e-02,  3.1950e-01,  1.3291e+00,  2.2020e+00,
           1.3006e+00,  3.1758e-01, -6.8640e-01,  5.6020e-01,  4.5558e-01,
           1.6782e+00,  3.5211e-01,  5.2138e-01,  5.4931e-01,  1.1859e+00,
          -6.2605e-01,  1.8730e+00, -5.2796e-01,  9.3002e-01, -3.0603e-01,
          -2.3061e-01,  1.3193e+00,  1.8163e+00,  2.2784e+00,  2.7020e+00,
          -1.1796e+00,  2.3714e+00, -1.9390e+00, -5.0095e-01,  6.3739e-01,
          -2.8435e+00,  4.5657e-01, -1.5366e+00, -1.1300e-01, -1.5641e+00,
          -4.9197e-01,  5.6484e-01, -2.0559e+00, -1.6089e+00, -2.4826e+00,
           3.4960e-01, -3.2278e-01,  4.1221e-02, -1.8241e+00,  3.6458e+00,
           1.0119e-01,  3.3301e-01, -2.7021e-01,  1.1265e+00,  9.2826e-01,
          -4.5826e-01, -5.7838e-01,  6.6694e-01,  8.9706e-01, -9.4149e-02,
           8.3338e-01, -1.9934e-01, -2.8021e+00,  2.3966e+00,  4.0510e-01,
           7.5744e-01,  5.2665e-01,  3.6013e+00,  2.7843e-01, -1.2669e+00,
          -7.0452e-01, -1.8148e+00, -4.8905e-01,  1.1644e+00,  9.6255e-01,
          -1.2303e-01,  3.8567e-01,  1.9121e-01, -1.2093e+00,  2.1243e+00,
          -1.8261e-02,  1.4402e-01,  1.4605e+00, -1.0605e+00,  3.2663e-01,
           9.8630e-01,  1.8650e+00, -2.1505e-02, -2.0255e+00, -9.9494e-01,
           8.3737e-02,  3.9045e-01,  1.5991e+00, -8.1499e-02, -2.0337e+00,
           4.7016e-01,  3.5955e+00, -9.3941e-01, -6.9175e-01,  1.6971e+00,
          -9.6395e-01,  2.2353e-01, -1.7541e+00, -6.7524e-01, -2.2726e+00,
           2.9339e+00,  1.5387e+00, -3.0159e+00, -7.8308e-03, -4.4138e-01,
           1.7583e-01,  2.0725e-01, -9.6595e-01,  8.7790e-01,  2.0177e+00,
          -1.1231e+00,  1.7540e+00, -3.6536e-02, -5.0842e-01, -1.4925e+00,
          -2.3393e-01,  2.4772e+00, -2.8856e+00, -1.5468e+00, -1.5484e+00,
          -8.2019e-01, -1.0824e+00,  7.9542e-01, -3.4348e-01, -1.1281e+00,
          -4.0897e-01,  1.6325e+00, -1.1989e+00,  1.1168e+00,  1.5069e-01,
           1.0875e-01, -3.5547e-02,  3.1278e-01,  1.6741e+00,  3.2415e+00,
          -2.1373e-01,  1.2138e+00,  5.4589e-01,  2.3086e+00,  2.0317e+00,
          -6.9705e-01, -2.1754e-01, -7.2256e-01,  4.5619e-01,  3.1298e-02,
           3.6782e-01,  8.8157e-01,  5.9555e-01, -1.3944e+00, -2.6676e-01,
          -1.6828e-01,  3.0656e-01, -7.6220e-01,  5.6292e-01, -4.0911e-01,
          -6.9269e-01,  6.5661e-01, -3.6243e+00, -9.7248e-01,  2.5631e-01,
           1.4970e+00,  1.4751e+00, -2.7577e+00,  7.5327e-01,  2.4463e-01,
           9.8457e-01,  1.0682e+00,  1.0074e+00, -2.9278e-01, -1.4238e+00,
          -4.3739e-01, -1.0756e+00, -4.8435e-02,  2.9760e-03, -1.0128e+00,
           9.8025e-01,  9.8648e-01, -3.1231e-01,  1.6458e+00, -5.1461e-01,
           8.2042e-01, -9.3869e-01, -2.0010e+00, -8.5610e-01, -1.3002e+00,
          -4.9204e-01,  1.7123e+00, -1.2003e+00, -2.0717e+00,  1.7454e+00,
          -1.0655e-01, -6.2619e-01,  2.0212e-01, -3.6761e-01,  6.5704e-01,
           4.0339e-01,  8.4627e-01, -1.6147e+00,  2.0114e-01, -1.0016e+00,
          -2.1337e+00, -9.2129e-01,  7.2167e-03, -6.1580e-01, -2.5847e-01,
          -2.7930e-01, -6.6958e-01,  6.2821e-01, -2.2891e-01, -4.5327e-01,
          -5.5012e-01,  3.5405e-01,  3.6797e-01, -1.4354e+00, -3.0881e+00,
          -2.1124e+00, -6.3526e-02, -8.8662e-01, -1.7588e+00, -1.5474e+00,
           2.0632e+00,  2.0842e-01, -9.2082e-01, -1.3020e+00, -9.8093e-03,
           2.2013e+00,  8.3966e-01, -2.3422e+00, -4.2742e-01, -5.9746e-01,
          -2.9574e-01, -9.3978e-01, -8.6268e-01,  7.6232e-01,  2.7702e+00,
           1.6468e+00,  4.6118e-01,  2.2467e+00, -7.4294e-01,  3.5219e+00,
           1.5808e+00,  3.3533e+00,  2.1643e+00, -2.0408e+00, -6.2057e-01,
           1.8434e+00,  1.8619e+00,  4.2796e-01, -3.9531e-01, -2.6533e+00,
           1.7230e+00,  3.5036e+00,  1.1856e+00, -2.7387e-01,  2.9428e+00,
           5.2435e-01,  6.3806e-02,  2.0564e-01, -4.4747e-01,  2.1786e+00,
          -1.8168e+00, -2.2135e-01,  1.8994e+00,  2.8691e+00,  1.0664e+00,
          -2.1949e-01, -6.8397e-01,  2.0690e+00,  1.4969e+00,  1.7710e+00,
           1.7672e+00,  1.1165e+00,  8.2307e-01,  4.4412e-02, -3.7833e-01,
           1.4870e+00,  1.9357e-01,  4.4427e-01, -1.2495e+00,  1.3334e+00,
           1.3215e-01, -7.6453e-01,  7.5182e-01,  9.0203e-01,  2.2762e+00,
          -6.5264e-01,  1.2479e+00, -1.5620e+00, -1.3051e-01,  8.6754e-01,
           1.0970e+00,  4.1571e-01,  2.8977e+00,  6.5338e-01, -1.9502e+00,
           2.5388e+00,  9.9000e-02, -2.7388e+00, -2.5102e+00, -1.2384e-01,
          -7.9847e-01,  2.0319e+00,  1.6983e+00,  1.1795e+00,  3.3911e+00,
           1.1443e+00,  5.2504e-01,  1.0879e+00,  1.5306e+00,  2.3714e+00,
           3.0553e+0###0, -9.8472e-01,  1.0600e+00, -4.7703e-02,  1.7101e+00,
          -1.2167e+00,  2.7219e+00,  1.0349e-01,  1.6886e+00,  1.7509e+00,
           9.5636e-01, -8.7598e-01, -4.0613e-01,  1.3070e+00, -3.5357e-01,
           5.3181e-01,  1.3287e+00, -1.2577e+00,  4.0429e-01, -1.6586e+00,
           8.0877e-01,  7.5141e-01,  1.9552e+00,  4.4665e+00, -8.7177e-01,
          -1.4115e+00,  2.3375e+00, -2.2267e+00,  2.0388e+00,  1.4562e+00,
          -2.1483e+00,  3.7177e+00,  3.8198e+00, -2.3819e-01, -8.5087e-01,
           2.0916e-01,  2.2043e+00,  9.2tensor([[[ 1.5926e+00,  1.4903e-01, -1.9363e+00, -4.8722e-01, -1.1463e-01,
          -3.1369e-01,  1.5872e-01, -8.5588e-02,  4.1568e-01,  6.0201e-01,
           1.8339e+00,  6.3042e-01, -9.0124e-01,  2.8473e+00,  1.3393e+00,
          -3.1858e-01, -1.4642e+00, -9.7470e-01, -5.8856e-02, -1.2062e+00,
           1.3936e+00, -2.3910e+00,  1.8439e-01,  1.5528e+00, -1.0435e-01,
           2.2393e-01,  6.4896e-01, -1.9608e+00,  3.4254e-02,  6.6968e-01,
          -1.7946e+00, -2.9223e-01,  1.4064e+00, -1.4172e+00, -1.3038e+00,
           4.0060e-01, -3.0584e-01, -4.7706e-01,  1.9176e+00,  5.6831e-01,
          -2.1380e+00, -1.3978e+00,  6.7678e-02,  8.4420e-01, -2.9817e-01,
           1.6259e+00, -7.8384e-01, -4.2055e-01,  4.5245e-01, -2.8765e-02,
           7.2769e-01,  1.6472e+00, -1.9210e-01, -1.5413e-01,  6.1298e-01,
          -2.0137e+00,  1.5595e+00,  2.9375e-01,  6.2810e-02, -1.2892e+00,
           2.4103e-01, -1.2357e-01,  1.3522e-01, -2.3203e-02, -1.3191e-01,
          -3.7480e-02, -4.6885e-01,  5.8088e-01,  1.7805e+00,  5.0462e-02,
          -2.1071e+00, -8.1988e-02,  8.7968e-01,  1.9173e-01,  1.0045e-01,
          -8.9672e-01,  5.0590e-01, -1.9896e+00,  4.7158e-01,  7.9897e-01,
           2.5534e-01,  8.9514e-01, -7.2745e-01,  1.4118e+00,  6.2158e-01,
           1.0473e+00, -9.8501e-01, -1.3891e+00, -1.7499e-01,  2.3006e+00,
           1.6944e+00, -1.0036e+00, -1.1281e+00,  4.5403e-01,  6.3459e-01,
          -1.0933e+00, -1.1945e-01, -1.9502e+00,  1.0385e+00, -5.1282e-01,
          -1.7444e+00,  5.6642e-02,  9.8907e-01,  1.1420e+00,  4.7963e-01,
           1.7119e+00, -9.7098e-01, -8.1925e-01,  1.8336e-01,  5.1169e-01,
          -1.6276e+00, -9.7969e-01, -1.2539e+00, -1.1820e-01,  4.3968e-02,
          -1.5266e-01,  8.5660e-01, -1.4707e+00, -4.4053e-01, -6.7992e-01,
          -1.1487e+00, -2.5162e+00, -2.3174e+00,  8.1140e-01,  3.3294e-01,
          -6.4979e-01, -1.7682e+00,  2.1292e-01, -9.8410e-01, -7.0048e-02,
          -1.1563e-01, -1.3907e+00, -1.0169e+00,  3.2415e-02, -1.5150e+00,
          -3.1091e-01,  1.7240e+00,  2.7174e-01, -1.2760e+00,  7.1867e-01,
           7.7223e-01, -7.8902e-01, -9.1196e-01,  2.5110e-01,  1.5351e+00,
          -4.5259e-01, -1.5686e+00,  9.4680e-01,  8.6220e-01,  5.1541e-01,
          -7.2627e-01, -6.0425e-02, -1.2908e+00, -1.2622e+00,  1.3498e-01,
           1.0486e+00, -1.2244e+00,  9.5673e-01,  8.0317e-01,  1.0481e+00,
           9.1044e-01, -5.4240e-01, -3.8804e-01,  6.8607e-01,  4.3172e-01,
          -3.8384e-01,  8.5490e-01, -3.3373e-01,  1.6545e+00,  1.5125e+00,
           5.5179e-01, -1.1720e+00,  4.0738e-01,  9.9197e-01,  3.5977e-01,
          -2.0415e-01, -3.8673e-01,  2.3899e-01,  1.1833e+00,  9.7554e-01,
          -1.2559e+00, -7.2660e-03, -8.4795e-02, -1.5282e+00, -9.9120e-01,
           1.6364e+00, -1.6038e+00,  5.7175e-01, -1.0211e-01, -1.1046e+00,
          -7.2209e-01,  1.6402e+00,  1.3649e-01, -1.0885e-01,  8.7585e-01,
           6.2614e-01, -1.7029e+00, -1.3403e+00,  6.1926e-01,  7.6595e-03,
          -1.1205e+00,  3.5420e+00,  9.1477e-02,  1.3405e+00, -1.2502e+00,
          -1.5520e-01,  1.4269e+00, -5.6524e-01,  6.4850e-02, -6.9822e-01,
           1.3282e+00,  5.3874e-01, -1.3544e+00,  3.1906e-01,  8.3440e-01,
           1.4471e-01,  1.3900e+00,  1.2772e+00,  1.1000e+00, -5.4269e-02,
           1.0679e+00,  6.0762e-02,  1.2609e+00,  9.7722e-01, -2.3929e-01,
           9.4800e-01,  1.3711e+00, -1.5812e+00, -1.6591e+00,  3.6821e-01,
          -7.3091e-01, -8.2048e-01, -4.5596e-01, -8.2925e-01, -2.4027e+00,
          -2.3991e+00, -1.3126e-01,  1.5423e-01, -7.8750e-01, -2.3205e+00,
          -3.3065e-01, -1.5721e+00, -4.3781e-02, -2.4833e-01, -1.0728e+00,
           8.6316e-01, -1.1219e+00, -9.1624e-01,  9.7516e-01,  2.1148e+00,
           1.1546e+00, -1.2716e+00, -1.7708e+00,  2.0544e+00, -1.0613e-01,
          -1.0242e+00, -1.7711e-01,  4.5789e-01,  3.0068e-02, -4.8188e-01,
           7.3381e-01,  8.4664e-01, -1.1324e+00,  1.1253e+00,  7.1618e-01,
          -1.0527e+00,  2.3005e+00,  7.5551e-01, -6.4637e-02,  3.8675e+00,
           2.7521e+00,  5.4619e+00, -4.0695e-01, -1.5720e+00, -6.0386e-01,
          -6.4378e-01,  6.0637e-01, -2.4860e+00,  2.6776e-01, -4.6756e-01,
          -9.2028e-01, -5.3512e-01, -1.7374e+00, -2.3594e+00,  2.8890e-02,
          -3.0363e-01,  1.7281e+00, -4.2819e-01, -2.4373e+00, -1.3590e+00,
          -8.4002e-01,  4.8848e+00, -2.0413e-01,  8.3749e-01, -5.9883e-02,
           1.0951e+00,  1.3701e+00,  9.5162e-01, -1.0910e-01, -1.5202e-01,
           1.9033e-01,  3.6762e-01, -2.3136e+00, -1.1187e+00,  9.0735e-01,
          -2.3286e+00,  5.7680e-02, -2.2097e-03, -4.3465e-01, -6.3537e-01,
          -8.6144e-01, -9.3189e-01, -3.7853e-02, -2.6083e+00,  1.2774e+00,
           8.7335e-02, -1.6886e-01, -5.8145e-01,  1.5501e+00, -1.3137e+00,
          -2.1320e+00,  3.9164e-01, -6.9570e-01,  5.5648e-01,  7.0010e-01,
           2.1371e+00,  7.6955e-01, -1.9051e+00,  1.1031e-01, -7.5243e-01,
           1.4055e-01, -3.4220e-01, -1.0158e+00,  1.5569e-01, -4.4148e-01,
           1.2958e+00,  1.1316e+00,  2.2123e+00,  1.4300e+00,  1.2077e-01,
          -5.5495e-01, -6.4004e-02,  5.6646e-01, -4.2922e-01, -2.9585e+00,
          -1.1811e+00, -3.9775e-01,  6.6341e-01, -1.3940e+00, -8.3236e-01,
          -3.1316e-02,  1.2354e+00, -2.0552e+00, -7.6575e-01, -6.0639e-02,
          -5.5849e-01,  9.8452e-01, -1.6545e+00,  2.7485e+00, -7.8059e-01,
           1.3168e+00,  2.3018e-01,  9.0090e-01,  1.0419e+00,  1.7032e-01,
          -1.2684e+00,  2.0124e+00,  2.3983e-01,  3.1269e+00, -3.1338e-01,
          -2.1044e-01, -6.3878e-01, -5.9498e-01, -7.2360e-01, -9.0680e-01,
          -4.5561e-01, -3.6730e-01, -2.2679e+00, -2.2163e+00, -1.9266e+00,
           5.9773e-01,  3.7412e-01,  1.2982e+00,  7.6588e-01,  2.0124e+00,
           5.7702e-02, -8.4128e-01,  6.5885e-01, -1.2354e-01,  6.7260e-01,
           4.5318e-01, -5.1718e-01, -7.3486e-01,  1.0526e+00,  2.1005e-01,
          -1.4181e+00, -7.5228e-01, -1.1830e+00, -5.7858e-01,  1.3914e+00,
           1.4739e-01,  1.6403e+00,  1.7473e+00,  1.1311e+00, -1.3211e+00,
          -6.0783e-01,  3.8356e-01,  1.7666e+00, -2.3250e+00,  4.8846e-01,
           1.8338e-01, -3.3265e-01, -1.7942e+00,  2.6059e-01,  8.4566e-01,
           1.7748e-01,  2.0274e+00,  7.4347e-01,  3.1729e+00, -1.0204e-01,
           2.5399e+00,  9.7689e-01, -1.6734e+00,  6.8471e-01,  2.4912e+00,
           3.1201e-02, -2.7474e-01, -2.2852e+00, -8.1141e-01,  1.5610e-01,
          -2.1275e+00, -1.0630e+00,  1.2871e-01,  5.3298e-01, -7.0833e-01,
           8.7788e-01,  2.5054e+00, -7.6257e-01, -2.7009e-01, -2.3443e+00,
          -1.6027e+00, -1.3413e+00, -5.4400e-01,  1.6610e-01, -8.2185e-01,
           6.4313e-01, -1.0709e+00, -3.0561e-01,  1.5037e-01, -2.0450e+00,
          -8.5557e-01, -1.5738e+00,  4.0416e-01,  3.5450e-01,  1.5143e+00,
          -1.8878e-02, -1.9515e+00,  4.7576e-01, -1.0465e+00,  1.7538e+00,
           1.8635e-01,  9.9753e-01, -1.3544e-01, -7.8783e-01,  1.2368e-01,
           8.9801e-01,  1.6109e+00,  5.1271e-01,  1.1708e+00, -1.0192e-01,
           1.8522e+00, -6.6407e-01, -9.6391e-01,  7.4491e-01,  1.4334e+00,
           2.0500e+00, -6.6480e-01,  8.5300e-01,  3.9791e-01,  3.8700e+00,
           7.0810e-02, -1.9291e+00, -7.1019e-01,  1.5276e+00, -3.9976e-01,
           9.2202e-01,  3.9650e-01,  5.6932e-04,  5.0626e-01,  4.5135e-02,
          -4.5675e-01, -8.4795e-01, -2.3929e+00,  9.7912e-01,  2.7922e+00,
           1.1368e+00,  2.3700e+00, -1.8097e+00, -1.4850e+00,  6.2625e-01,
           1.4358e+00,  1.0360e+00, -1.6741e+00,  1.6479e-01, -6.9482e-01,
          -7.1209e-02, -4.1835e-01,  8.9293e-01,  4.6286e-02,  9.2615e-01,
           2.0592e-01,  3.3684e-01]]], device='cuda:0')277e-01,  2.1049e+00,  2.0408e+00,
           4.5118e-01,  1.2441e+00,  1.9610e+00,  2.0923e+00,  1.0995e+00,
          -2.4803e-01,  1.2578e+00,  4.0394e+00,  8.1743e-01, -6.1004e-02,
          -1.4404e+00, -5.8319e-01,  1.9677e-01, -3.6442e-01,  7.4483e-01,
           2.6016e+00,  3.7177e+00,  2.6644e+00,  2.6548e+00,  1.3082e-01,
           6.5047e-01, -6.0697e-01, -1.3946e+00, -9.2804e-01, -2.9766e-01,
          -6.8708e-01,  2.5743e-02,  1.9384e+00,  1.7963e-01,  4.5725e-01,
           1.4271e+00,  5.7205e-01, -1.6023e+00,  2.8394e+00,  1.1833e+00,
          -2.4554e-02,  9.0153e-01, -1.6871e+00,  1.2551e+00, -8.6996e-01,
          -2.1120e+00,  4.4103e+00,  7.9482e-01,  5.2325e-01,  6.9466e-01,
           1.0924e+00,  4.2331e-01,  1.7tensor([[[ 1.5926e+00,  1.4903e-01, -1.9363e+00, -4.8722e-01, -1.1463e-01,
          -3.1369e-01,  1.5872e-01, -8.5588e-02,  4.1568e-01,  6.0201e-01,
           1.8339e+00,  6.3042e-01, -9.0124e-01,  2.8473e+00,  1.3393e+00,
          -3.1858e-01, -1.4642e+00, -9.7470e-01, -5.8856e-02, -1.2062e+00,
           1.3936e+00, -2.3910e+00,  1.8439e-01,  1.5528e+00, -1.0435e-01,
           2.2393e-01,  6.4896e-01, -1.9608e+00,  3.4254e-02,  6.6968e-01,
          -1.7946e+00, -2.9223e-01,  1.4064e+00, -1.4172e+00, -1.3038e+00,
           4.0060e-01, -3.0584e-01, -4.7706e-01,  1.9176e+00,  5.6831e-01,
          -2.1380e+00, -1.3978e+00,  6.7678e-02,  8.4420e-01, -2.9817e-01,
           1.6259e+00, -7.8384e-01, -4.2055e-01,  4.5245e-01, -2.8765e-02,
           7.2769e-01,  1.6472e+00, -1.9210e-01, -1.5413e-01,  6.1298e-01,
          -2.0137e+00,  1.5595e+00,  2.9375e-01,  6.2810e-02, -1.2892e+00,
           2.4103e-01, -1.2357e-01,  1.3522e-01, -2.3203e-02, -1.3191e-01,
          -3.7480e-02, -4.6885e-01,  5.8088e-01,  1.7805e+00,  5.0462e-02,
          -2.1071e+00, -8.1988e-02,  8.7968e-01,  1.9173e-01,  1.0045e-01,
          -8.9672e-01,  5.0590e-01, -1.9896e+00,  4.7158e-01,  7.9897e-01,
           2.5534e-01,  8.9514e-01, -7.2745e-01,  1.4118e+00,  6.2158e-01,
           1.0473e+00, -9.8501e-01, -1.3891e+00, -1.7499e-01,  2.3006e+00,
           1.6944e+00, -1.0036e+00, -1.1281e+00,  4.5403e-01,  6.3459e-01,
          -1.0933e+00, -1.1945e-01, -1.9502e+00,  1.0385e+00, -5.1282e-01,
          -1.7444e+00,  5.6642e-02,  9.8907e-01,  1.1420e+00,  4.7963e-01,
           1.7119e+00, -9.7098e-01, -8.1925e-01,  1.8336e-01,  5.1169e-01,
          -1.6276e+00, -9.7969e-01, -1.2539e+00, -1.1820e-01,  4.3968e-02,
          -1.5266e-01,  8.5660e-01, -1.4707e+00, -4.4053e-01, -6.7992e-01,
          -1.1487e+00, -2.5162e+00, -2.3174e+00,  8.1140e-01,  3.3294e-01,
          -6.4979e-01, -1.7682e+00,  2.1292e-01, -9.8410e-01, -7.0048e-02,
          -1.1563e-01, -1.3907e+00, -1.0169e+00,  3.2415e-02, -1.5150e+00,
          -3.1091e-01,  1.7240e+00,  2.7174e-01, -1.2760e+00,  7.1867e-01,
           7.7223e-01, -7.8902e-01, -9.1196e-01,  2.5110e-01,  1.5351e+00,
          -4.5259e-01, -1.5686e+00,  9.4680e-01,  8.6220e-01,  5.1541e-01,
          -7.2627e-01, -6.0425e-02, -1.2908e+00, -1.2622e+00,  1.3498e-01,
           1.0486e+00, -1.2244e+00,  9.5673e-01,  8.0317e-01,  1.0481e+00,
           9.1044e-01, -5.4240e-01, -3.8804e-01,  6.8607e-01,  4.3172e-01,
          -3.8384e-01,  8.5490e-01, -3.3373e-01,  1.6545e+00,  1.5125e+00,
           5.5179e-01, -1.1720e+00,  4.0738e-01,  9.9197e-01,  3.5977e-01,
          -2.0415e-01, -3.8673e-01,  2.3899e-01,  1.1833e+00,  9.7554e-01,
          -1.2559e+00, -7.2660e-03, -8.4795e-02, -1.5282e+00, -9.9120e-01,
           1.6364e+00, -1.6038e+00,  5.7175e-01, -1.0211e-01, -1.1046e+00,
          -7.2209e-01,  1.6402e+00,  1.3649e-01, -1.0885e-01,  8.7585e-01,
           6.2614e-01, -1.7029e+00, -1.3403e+00,  6.1926e-01,  7.6595e-03,
          -1.1205e+00,  3.5420e+00,  9.1477e-02,  1.3405e+00, -1.2502e+00,
          -1.5520e-01,  1.4269e+00, -5.6524e-01,  6.4850e-02, -6.9822e-01,
           1.3282e+00,  5.3874e-01, -1.3544e+00,  3.1906e-01,  8.3440e-01,
           1.4471e-01,  1.3900e+00,  1.2772e+00,  1.1000e+00, -5.4269e-02,
           1.0679e+00,  6.0762e-02,  1.2609e+00,  9.7722e-01, -2.3929e-01,
           9.4800e-01,  1.3711e+00, -1.5812e+00, -1.6591e+00,  3.6821e-01,
          -7.3091e-01, -8.2048e-01, -4.5596e-01, -8.2925e-01, -2.4027e+00,
          -2.3991e+00, -1.3126e-01,  1.5423e-01, -7.8750e-01, -2.3205e+00,
          -3.3065e-01, -1.5721e+00, -4.3781e-02, -2.4833e-01, -1.0728e+00,
           8.6316e-01, -1.1219e+00, -9.1624e-01,  9.7516e-01,  2.1148e+00,
           1.1546e+00, -1.2716e+00, -1.7708e+00,  2.0544e+00, -1.0613e-01,
          -1.0242e+00, -1.7711e-01,  4.5789e-01,  3.0068e-02, -4.8188e-01,
           7.3381e-01,  8.4664e-01, -1.1324e+00,  1.1253e+00,  7.1618e-01,
          -1.0527e+00,  2.3005e+00,  7.5551e-01, -6.4637e-02,  3.8675e+00,
           2.7521e+00,  5.4619e+00, -4.0695e-01, -1.5720e+00, -6.0386e-01,
          -6.4378e-01,  6.0637e-01, -2.4860e+00,  2.6776e-01, -4.6756e-01,
          -9.2028e-01, -5.3512e-01, -1.7374e+00, -2.3594e+00,  2.8890e-02,
          -3.0363e-01,  1.7281e+00, -4.2819e-01, -2.4373e+00, -1.3590e+00,
          -8.4002e-01,  4.8848e+00, -2.0413e-01,  8.3749e-01, -5.9883e-02,
           1.0951e+00,  1.3701e+00,  9.5162e-01, -1.0910e-01, -1.5202e-01,
           1.9033e-01,  3.6762e-01, -2.3136e+00, -1.1187e+00,  9.0735e-01,
          -2.3286e+00,  5.7680e-02, -2.2097e-03, -4.3465e-01, -6.3537e-01,
          -8.6144e-01, -9.3189e-01, -3.7853e-02, -2.6083e+00,  1.2774e+00,
           8.7335e-02, -1.6886e-01, -5.8145e-01,  1.5501e+00, -1.3137e+00,
          -2.1320e+00,  3.9164e-01, -6.9570e-01,  5.5648e-01,  7.0010e-01,
           2.1371e+00,  7.6955e-01, -1.9051e+00,  1.1031e-01, -7.5243e-01,
           1.4055e-01, -3.4220e-01, -1.0158e+00,  1.5569e-01, -4.4148e-01,
           1.2958e+00,  1.1316e+00,  2.2123e+00,  1.4300e+00,  1.2077e-01,
          -5.5495e-01, -6.4004e-02,  5.6646e-01, -4.2922e-01, -2.9585e+00,
          -1.1811e+00, -3.9775e-01,  6.6341e-01, -1.3940e+00, -8.3236e-01,
          -3.1316e-02,  1.2354e+00, -2.0552e+00, -7.6575e-01, -6.0639e-02,
          -5.5849e-01,  9.8452e-01, -1.6545e+00,  2.7485e+00, -7.8059e-01,
           1.3168e+00,  2.3018e-01,  9.0090e-01,  1.0419e+00,  1.7032e-01,
          -1.2684e+00,  2.0124e+00,  2.3983e-01,  3.1269e+00, -3.1338e-01,
          -2.1044e-01, -6.3878e-01, -5.9498e-01, -7.2360e-01, -9.0680e-01,
          -4.5561e-01, -3.6730e-01, -2.2679e+00, -2.2163e+00, -1.9266e+00,
           5.9773e-01,  3.7412e-01,  1.2982e+00,  7.6588e-01,  2.0124e+00,
           5.7702e-02, -8.4128e-01,  6.5885e-01, -1.2354e-01,  6.7260e-01,
           4.5318e-01, -5.1718e-01, -7.3486e-01,  1.0526e+00,  2.1005e-01,
          -1.4181e+00, -7.5228e-01, -1.1830e+00, -5.7858e-01,  1.3914e+00,
           1.4739e-01,  1.6403e+00,  1.7473e+00,  1.1311e+00, -1.3211e+00,
          -6.0783e-01,  3.8356e-01,  1.7666e+00, -2.3250e+00,  4.8846e-01,
           1.8338e-01, -3.3265e-01, -1.7942e+00,  2.6059e-01,  8.4566e-01,
           1.7748e-01,  2.0274e+00,  7.4347e-01,  3.1729e+00, -1.0204e-01,
           2.5399e+00,  9.7689e-01, -1.6734e+00,  6.8471e-01,  2.4912e+00,
           3.1201e-02, -2.7474e-01, -2.2852e+00, -8.1141e-01,  1.5610e-01,
          -2.1275e+00, -1.0630e+00,  1.2871e-01,  5.3298e-01, -7.0833e-01,
           8.7788e-01,  2.5054e+00, -7.6257e-01, -2.7009e-01, -2.3443e+00,
          -1.6027e+00, -1.3413e+00, -5.4400e-01,  1.6610e-01, -8.2185e-01,
           6.4313e-01, -1.0709e+00, -3.0561e-01,  1.5037e-01, -2.0450e+00,
          -8.5557e-01, -1.5738e+00,  4.0416e-01,  3.5450e-01,  1.5143e+00,
          -1.8878e-02, -1.9515e+00,  4.7576e-01, -1.0465e+00,  1.7538e+00,
           1.8635e-01,  9.9753e-01, -1.3544e-01, -7.8783e-01,  1.2368e-01,
           8.9801e-01,  1.6109e+00,  5.1271e-01,  1.1708e+00, -1.0192e-01,
           1.8522e+00, -6.6407e-01, -9.6391e-01,  7.4491e-01,  1.4334e+00,
           2.0500e+00, -6.6480e-01,  8.5300e-01,  3.9791e-01,  3.8700e+00,
           7.0810e-02, -1.9291e+00, -7.1019e-01,  1.5276e+00, -3.9976e-01,
           9.2202e-01,  3.9650e-01,  5.6932e-04,  5.0626e-01,  4.5135e-02,
          -4.5675e-01, -8.4795e-01, -2.3929e+00,  9.7912e-01,  2.7922e+00,
           1.1368e+00,  2.3700e+00, -1.8097e+00, -1.4850e+00,  6.2625e-01,
           1.4358e+00,  1.0360e+00, -1.6741e+00,  1.6479e-01, -6.9482e-01,
          -7.1209e-02, -4.1835e-01,  8.9293e-01,  4.6286e-02,  9.2615e-01,
           2.0592e-01,  3.3684e-01]]], device='cuda:0')037e+00,  1.6165e+00, -1.5474e-02,
           1.7080e+00,  1.0196e+00, -2.4621e-01,  3.2248e+00, -2.5169e-01,
          -3.0593e-01,  5.3668e-01,  2.2985e+00, -1.4582e+00, -5.1559e-01,
           1.6194e+00,  9.3175e-01,  9.1110e-02,  8.0237e-01,  2.3372e+00,
          -3.0931e-02,  2.0398e+00,  2.1281e+00, -8.1375e-01,  3.4531e+00,
           3.6994e+00,  1.6856e+00, -9.7553e-01,  3.4849e+00,  1.5267e+00,
          -1.8112e+00,  5.7375e-01, -6.0930e-01,  1.1341e+00,  3.0162e+00,
           5.6020e-01, -2.1561e-01, -4.5898e-01,  3.0886e+00,  1.1680e+00,
          -6.8300e-01,  1.3459e+00,  4.0179e+00,  2.6109e+00,  1.1050e+00,
           7.3527e-01,  9.6121e-01, -1.0022e-01, -8.5127e-01,  1.6260e+00,
          -1.0697e-01, -1.8565e-01, -1.5985e+00,  3.2789e+00,  6.2383e-01,
           9.6515e-01, -2.1720e+00,  2.6375e-01,  6.0737e-01,  1.6397e+00,
           2.8799e+00,  8.5449e-01, -2.6702e-01,  9.4815e-01,  1.0076e-01,
          -3.0805e-01, -9.3650e-01, -8.8185e-01, -6.9241e-01,  1.9024e+00,
           2.7826e+00,  1.7708e-01]]], device='cuda:0')
the value of enc: tensor([[[ 0.3330, -0.3364,  0.3980,  ...,  0.0761,  0.4965,  0.0103]],

        [[-0.3485, -0.2252,  0.4078,  ..., -0.0329, -1.6118, -0.6042]],

        [[-0.1036,  0.0610, -0.1291,  ...,  0.0070, -0.2148, -0.0408]]],
       device='cuda:0')
the value of padding mask: tensor([[False, False, False]], device='cuda:0')
the self attn mask: None
the self attn padding mask: None




final x output from the actual decoded sentence:

tensor([[[ 4.1598e-01,  6.0815e-01,  2.9221e-01, -3.6554e-01, -5.9126e-01,
           6.3717e-02,  3.3438e-01,  8.9581e-01, -1.0472e-01, -3.3786e-01,
           6.0666e-01,  1.1340e+00,  2.7670e-01,  5.2166e-01,  7.5971e-01,
          -2.6695e-01, -4.7553e-01,  2.3991e-01, -5.2322e-01,  3.2185e-01,
           1.2992e+00, -6.0927e-01, -2.6094e-01,  3.3353e-01, -7.0580e-01,
           2.4687e-01, -5.0183e-01, -5.7951e-01, -9.9759e-01,  7.3877e-01,
          -1.3239e+00,  3.8449e-01,  1.3994e+00, -2.2091e-01, -7.2065e-01,
          -1.2465e+00,  8.8378e-02, -1.7414e+00,  3.0108e-01, -1.3990e-01,
          -9.5436e-01,  1.2309e-01, -4.3137e-01, -4.2262e-01, -5.2017e-01,
           4.9245e-01, -1.7386e-01,  1.1155e-01, -3.0993e-02,  1.6612e-01,
           5.6253e-01,  7.4075e-01,  8.3604e-02,  8.9553e-01, -7.6408e-01,
          -6.7532e-02, -3.5491e-01, -5.6433e-01, -4.2609e-01,  7.3228e-01,
          -1.1929e+00,  4.8968e-01, -1.1768e+00,  1.1652e+00,  5.8667e-01,
          -6.4228e-01, -1.2807e-01,  5.5512e-01, -1.0054e-01,  3.8499e-01,
          -3.4458e-01, -4.3728e-01,  1.2696e-02, -2.0511e-02, -1.2818e-01,
          -8.9327e-01, -1.0032e+00,  3.1581e-01,  2.7821e-01, -8.9873e-01,
           1.2289e-01,  1.6420e-02,  2.9626e-01, -1.3961e+00,  1.7562e-01,
           5.7389e-01, -5.4384e-01,  2.5767e-01, -2.2099e-01,  1.7212e+00,
           8.2362e-02,  3.2298e-01, -3.5968e-01,  3.2320e-01,  4.5957e-01,
           9.2758e-02,  3.9166e-01, -4.2184e+00,  9.3550e-01,  5.8330e-01,
          -8.8753e-01, -5.6671e-01,  4.4659e-01, -2.1165e-02,  1.4019e+00,
           1.3972e-01,  3.7193e-01,  8.3289e-01,  1.1327e+00, -5.4281e-01,
          -1.2457e+00, -4.3070e-01, -3.0695e-01, -2.5050e-01, -1.9668e-02,
          -1.0639e-01, -1.2176e-01, -1.1515e+00,  1.4256e-01, -1.8559e-01,
           6.1841e-01,  5.0376e-01, -2.2792e-02,  1.3876e+00,  3.3317e-01,
           1.3627e+00, -5.1208e-01, -1.3557e-01, -1.0502e+00, -9.9725e-01,
           1.6208e-01,  2.7780e-02,  7.9289e-02, -1.0756e+00, -2.1436e-02,
           1.2834e+00,  3.1982e-01,  3.1327e-02,  6.0082e-02,  7.3125e-01,
          -1.1674e-01, -7.6006e-01, -5.8706e-01,  3.2584e-01, -1.1583e-01,
           1.5262e+00, -1.3362e+00,  1.9707e-01, -1.2012e-01,  1.5424e-01,
           6.3950e-01,  4.2290e-01,  7.9167e-01,  8.5180e-01, -3.7747e-01,
           3.0010e-01,  2.0985e-01,  1.1340e-02,  7.8551e-01,  4.6892e-01,
           1.2581e+00,  1.1534e+00,  3.5721e-01, -6.9231e-01,  4.3860e-02,
           4.3962e-02, -2.8496e-01, -8.1263e-01, -1.4875e+00, -3.2513e-01,
          -3.6098e+00, -2.6349e-01,  4.7443e-02,  4.8898e-01,  4.5732e-01,
          -1.1205e+00,  2.0028e-01, -8.7137e-01, -5.8225e-02,  2.5042e+00,
          -4.1914e-01, -4.9137e-01, -3.6338e-01,  1.8517e-01,  8.9835e-01,
           9.0030e-01, -3.2864e-01, -6.9663e-01,  7.2208e-01, -6.0547e-01,
          -1.8666e-01,  1.5103e+00,  5.4958e-01, -5.7556e-01, -2.1508e-01,
          -2.3632e-01, -2.0295e-01, -6.1944e-01,  1.0427e+00,  8.7992e-01,
          -5.9441e-01,  3.8589e+00, -1.6607e+00, -1.3930e-01, -2.4646e-01,
          -2.1801e-01,  3.4940e-01, -1.2485e-01,  5.1405e-01, -9.1840e-01,
           9.1643e-01,  3.2019e-01,  2.1191e-01,  2.0722e-01, -8.1369e-01,
          -1.4867e-01, -2.6578e-01,  6.7207e-01,  1.2343e+00, -4.8023e-01,
           8.3520e-01,  3.2059e-01,  2.3837e-01,  1.9238e-01, -3.7754e-01,
          -9.8231e-02, -9.7218e-01, -3.7673e-02,  5.5573e-01,  7.4389e-01,
           1.7993e-01,  9.6113e-01, -8.3242e-02,  6.7258e-01, -6.7683e-01,
          -6.9109e-01, -9.0854e-01, -1.8125e-02, -3.2548e-01,  4.7027e-01,
          -1.0882e-01, -3.4528e-01, -8.4484e-01, -2.6589e-01,  3.3165e-01,
          -3.0412e-01, -9.0533e-01, -2.7115e-01, -5.6139e-01,  1.1451e+00,
          -5.3241e-01, -3.6114e-01, -6.0936e-02,  6.9256e-01, -7.0474e-01,
          -3.0340e-01, -3.1333e-02,  2.0477e-01,  2.1698e-01, -1.0363e-02,
          -1.3507e+00, -7.5281e-02,  9.9089e-01,  1.5860e+00, -6.0939e-01,
          -6.0933e-02,  3.4964e-01,  4.4606e-01, -5.9274e-04,  4.1803e-01,
           1.1145e+00,  2.6067e-01, -1.5776e+00, -1.7917e-01, -8.9134e-02,
          -8.6335e-01, -2.3252e-01,  1.4847e-01,  4.1869e-01,  1.2252e-01,
           2.9425e-01, -5.2194e-01, -1.8679e-01, -1.0075e-01,  3.1771e-01,
           6.6639e-01, -4.0957e-01,  1.2164e+00, -8.8143e-01, -2.5343e-01,
           6.8848e-01,  7.3516e-01,  4.1833e-01, -8.1404e-01, -8.4771e-01,
           1.1830e+00,  1.9878e+00, -1.7328e+00, -8.9842e-01,  6.4162e-01,
           4.1844e-01, -3.5750e-01, -5.9668e-02,  5.9375e-01,  7.4266e-01,
          -3.8122e-01,  1.9913e-01,  9.6805e-01,  1.7856e+00, -1.5956e-01,
           6.2723e-01,  2.7500e-01,  4.7528e-01, -2.2067e-02,  2.0914e-01,
           5.5715e-01,  5.4579e-01, -6.1412e-01, -4.5708e-01, -2.2894e-01,
          -3.1262e+00, -3.8854e-01,  8.9459e-02, -9.9137e-01,  5.8441e-01,
           1.8450e-02,  1.7461e-01, -4.0324e-02,  8.4853e-01,  1.0983e-01,
          -1.9269e-01,  5.4114e-01, -8.2449e-01,  1.4176e+00,  1.8520e-01,
           1.7135e+00, -6.3067e-02,  1.2737e+00, -5.9316e-02, -2.9060e-01,
           4.6146e-01, -3.2788e-01, -6.8265e-01, -1.1321e+00, -1.5050e+00,
          -9.6385e-01,  2.2193e-01,  8.0092e-01, -2.9867e-01,  1.0819e+00,
          -2.9339e-01,  3.9021e-01,  3.4720e-01, -5.4695e-01,  1.5046e+00,
          -2.4109e+00, -2.4567e-01,  1.4724e+00,  2.9109e-01,  7.0155e-01,
           7.7437e-01,  8.4727e-01,  3.9349e-01,  2.1057e+00, -9.9895e-01,
          -3.4418e-01, -4.6686e-01, -1.3358e+00,  1.5478e+00, -1.6552e-01,
          -9.2173e-01,  1.0886e+00, -1.1842e+00, -1.0283e+00, -6.4487e-01,
          -5.6387e-01, -1.1261e+00,  4.8820e-01, -2.9946e-01, -1.0998e+00,
          -1.7909e-01, -8.1311e-03,  6.7446e-01,  3.0340e-02,  1.1002e+00,
          -1.7216e+00,  8.2718e-01,  6.5188e-01,  8.1998e-01,  6.6652e-01,
          -8.8039e-01,  9.6393e-01,  4.4226e-01,  1.9228e-01, -7.9393e-01,
           6.4348e-01,  1.0173e+00,  2.1399e-01, -4.4376e-03,  1.2417e+00,
           9.8879e-02,  2.5464e-01,  6.4954e-01,  1.7655e+00, -7.1698e-01,
           2.0289e-01, -1.3504e+00,  7.7923e-01, -5.1765e-02,  8.2262e-01,
           6.5813e-01, -1.8861e-01, -7.5891e-01, -1.4136e-01,  5.6198e-01,
           2.7007e-01,  6.8265e-01,  4.1230e-01, -1.3737e-01,  5.9920e-01,
           6.8057e-01, -1.6940e-01,  3.7263e-01, -1.6920e-01,  8.0920e-01,
          -7.6046e-01, -6.8299e-01, -1.3929e+00,  1.2445e-01,  4.5652e-01,
          -6.8614e-01, -5.2800e-01, -7.9685e-01,  1.0619e+00, -8.4653e-01,
           4.2371e-01,  5.8421e-01, -1.3312e+00, -1.3850e+00, -1.0721e+00,
           1.2710e-01,  2.9257e-01,  2.7641e-01,  2.5436e+00, -9.4229e-01,
           5.4825e-01, -6.3899e-01, -8.6066e-01,  3.7768e-01,  4.1830e-01,
          -6.5427e-01,  4.3986e-01, -4.5040e+00, -1.7064e-01,  1.3174e+00,
           2.1591e-01, -5.9373e-02, -3.8366e-01,  1.4751e-01,  1.8545e+00,
          -1.3076e+00,  1.8883e+00,  7.6750e-01, -1.9271e-01, -3.1373e-01,
           4.6077e-01,  5.2284e-01,  4.8753e-01,  1.0783e+00, -8.4272e-03,
           6.1550e-01, -6.1735e-01,  1.0282e-01,  7.5129e-01,  7.1686e-02,
           4.9580e-01, -2.3906e-01,  4.7437e-01,  1.2877e+00,  5.8551e-01,
          -1.9858e-01, -2.7003e-01, -9.6462e-03,  6.1437e-01, -7.5688e-01,
          -1.8409e-01,  7.7206e-03,  7.2350e-01,  1.9674e-01,  5.0926e-01,
          -4.2825e-01, -2.9546e-01, -1.4323e+00,  2.1420e-01,  1.2253e+00,
          -4.8917e-01,  3.7088e-01,  7.2222e-01, -5.6242e-01,  2.1709e-01,
           7.1613e-01,  2.9016e-01,  1.4359e-01,  7.4340e-01,  1.0968e+00,
          -4.4245e-01, -1.3914e-01, -2.4115e-01, -1.7507e+00,  1.1720e+00,
           1.8348e-01,  7.3707e-01]]], device='cuda:0')




final x output from the prefix-constrained decoded sentence:
tensor([[[ 6.3712e-01,  4.4577e-01,  1.6205e-01, -8.1823e-01, -3.0059e-02,
          -1.2274e+00, -1.2875e+00,  1.9662e+00,  7.6473e-01, -9.3440e-02,
          -3.1825e-02,  1.5951e+00,  1.5145e+00,  4.6917e-01,  1.8781e+00,
          -1.2456e+00,  1.1340e-02, -2.0221e+00, -1.9288e-01,  6.9407e-01,
          -1.0687e+00, -1.3141e+00,  7.2066e-01, -5.9425e-01, -2.2042e+00,
          -5.4063e-01,  5.0985e-02, -8.0966e-01, -1.8362e+00,  1.2320e+00,
           6.1297e-01, -3.7252e-01,  9.1420e-01, -6.2792e-01, -1.8742e+00,
          -2.2290e-01,  2.0521e-01, -1.5641e+00,  7.2286e-01,  8.6511e-01,
          -1.7084e+00, -3.3163e-01, -1.7200e-01, -6.8119e-01, -1.0053e+00,
           1.2774e+00, -2.8919e-01, -1.3676e-01,  3.2027e-01,  5.4186e-01,
          -8.7616e-01, -6.6119e-01,  4.7561e-01,  1.1753e+00,  3.7225e-01,
           1.0000e+00, -4.5603e-01, -9.7651e-01, -1.0321e+00,  6.5159e-01,
          -1.4052e+00,  3.1032e-01, -6.6969e-01,  2.6201e+00, -2.5951e-01,
          -5.8945e-01,  1.9144e+00,  3.0096e+00,  2.1258e+00,  2.2250e-01,
          -1.3540e+00, -4.2846e-01, -2.4837e-01,  1.0034e+00, -5.0838e-01,
          -1.4718e+00,  4.8995e-01,  1.2819e+00,  1.5261e-01,  9.5558e-01,
          -4.2416e-01,  1.2023e+00,  3.8479e-01, -1.6120e+00,  1.1525e+00,
           1.0169e+00, -5.9710e-01,  1.0299e+00,  1.0033e+00,  1.1722e+00,
          -4.3318e-01,  2.5762e-01, -1.0012e+00, -4.9275e-02,  1.0871e+00,
          -7.0325e-01,  1.2236e+00, -3.4929e+00,  2.0783e-01,  5.1484e-02,
          -4.3534e-01, -1.9440e-01,  8.2600e-01, -7.4273e-01, -1.1529e+00,
          -8.3149e-01,  1.6294e-01,  6.6573e-01,  2.2870e-01,  9.9310e-01,
          -7.2456e-01, -8.0624e-01,  3.9901e-01, -4.2078e-01,  7.8937e-01,
           4.6894e-01,  1.1119e+00,  1.7692e-01,  6.6380e-01, -1.5969e+00,
           3.1723e-01, -1.0876e-01,  2.5568e-01,  1.7476e+00, -1.1354e+00,
           7.0782e-01,  4.3132e-01, -1.5620e-01, -3.6261e-01, -4.8451e-02,
           7.7895e-01,  6.4209e-01,  1.8830e-01, -7.8187e-01, -2.1239e-01,
           7.6706e-01,  1.2468e+00,  1.7663e+00, -1.3304e-02,  1.4464e+00,
          and constrained: -3.0569e-04, -1.1601e+00,  1.0736e+00, -1.2935e+00, -3.2795e-01,
           1.2744e+00, -1.0999e+00,  3.9795e-01,  6.3260e-01, -3.0463e-01,
          -4.3965e-02,  2.5657e-01, -8.0498e-01,  1.8130e+00, -8.9450e-01,
          -1.1116e+00,  1.4997e+00,  2.8569e-01,  5.4505e-02, -1.7186e-01,
           6.5745e-01,  3.1916e-01,  2.8572e-01,  6.0986e-01,  3.7796e-01,
           4.8662e-01,  1.5674e+00, -1.5654e+00,  7.3765e-01,  8.0146e-01,
          -2.6237e+00,  5.8377e-01, -1.2679e-01, -6.5834e-01,  5.9683e-01,
          -1.3595e+00, -1.4042e+00, -4.3305e-01, -2.4035e-01,  2.4419e+00,
          -6.3654e-01,  6.3101e-02, -1.0380e-01, -2.7979e-01,  9.0198e-01,
           1.3118e+00, -5.2506e-01, -8.9135e-01,  9.5349e-01, -2.2748e-01,
          -1.0003e-01,  2.0058e-01, -7.1888e-01, -1.4733e+00, -1.5949e-01,
           5.9776e-01, -9.3817e-02,  4.9600e-01, -4.9975e-01,  1.2863e+00,
           5.4577e-01,  2.9091e+00, -2.0445e+00,  8.0437e-01, -5.8571e-01,
          -8.5597e-01,  3.4161e-01, -6.7291e-01, -1.6314e-01,  8.8856e-01,
          -4.2947e-01, -3.9108e-01, -4.1613e-01, -9.0142e-01,  4.3730e-01,
          -2.4540e+00, -7.4888e-01,  4.3245e-01,  9.3668e-02, -2.7755e-01,
          -2.0236e-01, -5.8243e-01,  6.5754e-01, -7.3761e-01, -3.3439e-01,
           5.4879e-01,  4.9426e-01,  1.9005e+00, -4.8986e-01, -4.3208e-01,
           3.8433e-01, -7.4630e-01, -2.5263e+00,  3.4473e-01, -8.1923e-01,
          -2.5683e-01, -7.6223e-01, -1.2588e+00, -1.5461e+00,  7.5074e-01,
           3.5040e-01,  1.8542e-01, -7.4796e-01, -8.3253e-01,  3.4572e-01,
           8.0762e-01,  8.7578e-02,  1.3922e-02,  1.2798e+00,  6.8467e-01,
           1.7185e+00, -1.9148e+00,  2.3729e-01, -5.0584e-01, -1.1899e+00,
           1.5701e+00, -2.2360e-01,  7.8229e-01,  1.9412e-01,  4.9198e-01,
           7.4558e-02,  6.3851e-01,  1.9296e+00,  3.8307e-01, -1.6383e-01,
          -8.2513e-01,  2.0991e+00, -3.0529e-01,  4.8326e-01,  2.5633e-01,
           1.9064e+00,  5.1200e-01,  2.6397e-02, -3.6494e-01, -2.0331e+00,
          -7.4754e-01,  3.2403e-01,  1.9957e-01,  1.4719e+00, -1.1527e+00,
           5.1521e-01, -4.5728e-01, -1.6300e+00, -1.9061e-01,  8.3387e-01,
          -1.7755e+00,  1.2928e-01,  6.2335e-01, -5.9032e-01, -3.7503e-01,
          -4.2574e-01, -2.8873e-01, -4.8501e-01, -1.2479e+00, -1.1706e+00,
           6.5468e-01,  2.0806e+00, -1.8283e+00, -2.1381e-01, -2.7000e-01,
           2.5766e-02, -1.1901e+00,  3.3732e-01,  1.2348e+00,  5.8650e-01,
          -1.0209e+00,  5.1881e-02,  8.8585e-01,  6.9968e-01,  6.4668e-01,
           9.0367e-01, -4.1675e-01,  4.1037e+00,  9.3390e-01, -3.1050e-01,
          -1.4427e+00, -2.2902e+00,  2.0825e+00, -5.7929e-01, -2.8121e-02,
          -1.3058e+00,  5.8478e-01,  4.4012e-02, -2.1262e-01, -1.6860e+00,
          -2.8906e-01,  2.6913e-01,  3.3414e-02,  7.4944e-01,  1.0924e+00,
           1.0453e+00,  7.2533e-01, -5.6294e-01,  1.1819e-01, -3.7433e-01,
           1.6548e+00, -6.0793e-01,  9.5732e-01, -2.4538e-02,  1.5726e+00,
           2.4633e+00, -8.7476e-01, -9.5735e-01,  5.7715e-01, -1.1554e-01,
          -8.2911e-01,  1.3669e+00,  9.5578e-01, -9.4021e-01, -2.3846e-01,
           3.5837e-01, -7.2945e-01,  2.2046e+00, -8.2364e-01,  1.4911e+00,
          -2.6901e+00, -1.9816e+00,  1.4276e+00,  1.1109e+00, -3.4953e-02,
           1.1427e+00,  1.3981e+00,  1.9895e+00,  7.7071e-01,  3.4978e-01,
          -4.2396e-01, -1.3860e+00, -5.3272e-01,  5.5514e-01, -9.6569e-01,
          -1.3255e+00,  2.4626e-01,  5.2685e-02, -7.5902e-01,  1.2812e+00,
          -1.0287e+00, -3.8346e-01, -1.2546e-01, -2.7417e-02, -1.7826e+00,
          -2.4767e-01,  5.9505e-01,  7.2062e-02,  8.0803e-01, -1.6696e-01,
          -1.5872e+00,  9.3253e-01,  1.3105e-01,  2.6232e+00,  2.4969e-01,
           2.2176e-01,  2.0223e+00, -2.6434e-01, -5.3840e-01, -1.3425e+00,
           8.6953e-01,  7.4921e-01, -4.6300e-01,  1.5609e+00,  1.2750e+00,
          -1.8949e+00, -2.9709e-02,  1.8028e-01,  4.3929e-01,  6.8964e-01,
          -6.1832e-01, -1.1266e+00, -1.3412e+00, -7.7537e-01,  1.3740e+00,
           2.2468e+00,  7.1186e-01, -2.1673e+00, -5.1090e-01,  1.2522e+00,
           6.8648e-01,  1.3313e-01,  1.4885e+00, -2.5132e-01, -4.2033e-01,
          -1.1976e+00,  1.6958e-01, -1.7237e-01, -1.0152e+00,  1.3151e+00,
          -9.5640e-01, -1.4090e+00,  1.2560e+00, -6.7786e-01,  6.0476e-01,
           3.0716e-01,  3.1198e-01, -9.0389e-01,  1.0253e-01, -3.0242e+00,
           9.4542e-01,  1.5160e+00, -7.4469e-01,  1.7663e+00, -6.1771e-01,
          -9.1991e-01, -2.2687e-01,  9.3279e-01,  2.4227e+00, -1.5409e+00,
          -3.0198e-01, -1.6153e+00, -7.3144e-01, -7.7707e-01,  6.7343e-01,
          -6.0820e-01,  1.2132e+00, -3.1449e+00, -7.7793e-01, -8.7831e-01,
          -2.6412e-01, -4.2350e-01,  1.0036e-01,  5.3725e-01, -2.9902e-01,
          -1.1791e+00,  1.7430e+00, -1.6669e-01,  9.4401e-01, -7.9919e-01,
           3.5576e-01,  8.1187e-01,  9.7989e-01,  1.6819e+00, -3.7383e-01,
           8.2062e-01, -8.4307e-01, -2.5276e-01,  4.0520e-01,  9.9294e-01,
          -1.6172e-01, -1.3298e+00, -4.8713e-02, -4.0171e-01,  1.4219e+00,
           9.1482e-01, -1.6341e+00, -3.5699e-01,  3.1961e-01, -5.8499e-01,
           1.4786e-03, -2.1891e-01,  1.0991e+00, -7.3844e-01, -2.7663e-01,
           7.9174e-01,  4.7405e-01, -2.2417e+00,  1.8827e-01,  4.2518e-01,
           5.8847e-01, -8.8287e-01, -2.3723e-02,  2.9284e-01, -3.0746e-01,
           2.0851e+00, -5.6765e-01, -2.8720e+00, -4.4573e-02,  9.1576e-01,
           8.2588e-01,  1.2506e+00,  3.5438e-01, -4.9149e-01,  4.5914e-01,
           1.0637e-01,  6.4695e-02]]], device='cuda:0')


For the prefix constrained one we have 6 layers and on the final one we enter the layer_attn is not None and idx == alignment

For the actual one we have 6 layers and also enter the same at idx 5 


Now we are checking the inner states:

#### Prefix constrained ####
the x from the 0 step: tensor([[[-2.5521e-01, -1.8903e-01,  2.2783e-01,  4.0848e-01, -1.2613e+00,
          -2.2867e-01,  6.1063e-01, -3.8975e-03, -3.8397e-01, -3.9728e-01,
          -1.3308e+00, -3.8558e-01,  1.9030e+00, -2.0156e+00,  6.0013e-01,
          -5.5542e-01,  6.1803e-01, -5.5762e-01, -1.3710e+00,  4.7346e-01,
           7.4066e-01, -1.1956e-01, -2.9569e-01, -1.2337e+00, -6.4043e-02,
          -5.7733e-01, -8.6562e-01, -1.7343e+00,  6.3729e-01, -4.3207e-02,
          -6.2296e-01,  1.7472e-01,  2.4072e-03, -3.4544e-01, -8.3227e-01,
           6.4360e-01, -5.4118e-01, -6.2064e-01, -1.7711e-02, -8.3392e-01,
          -5.5006e-01, -1.0630e-01,  2.3713e+00,  2.5977e-02,  1.1285e+00,
          -1.8572e-01,  1.3772e-01, -8.8995e-02, -2.6230e-02,  6.9917e-01,
          -5.1892e-01,  9.0682e-01, -9.1402e-01,  6.5401e-01, -2.2936e-01,
           9.9321e-01, -7.1481e-01, -5.2960e-01, -1.2297e-01,  2.1316e-01,
          -4.3466e-02,  1.7539e+00, -1.3586e+00,  9.2037e-01,  8.5669e-02,
           4.7812e-01,  1.1823e+00, -3.6274e-01,  2.1835e-01,  9.4840e-01,
          -1.2634e+00,  1.0606e+00, -1.0910e+00,  8.4225e-01, -1.2333e-01,
          -9.5386e-01, -8.4413e-01, -3.8230e-01, -5.7025e-01, -5.9978e-01,
           2.7194e-01, -6.7479e-01, -8.2031e-01, -1.3598e+00, -7.7376e-01,
           2.7872e-01, -1.3107e+00, -3.7404e-01, -7.3343e-01,  9.0808e-01,
          -2.7085e-01,  5.5704e-01, -9.5301e-01,  7.4010e-01,  1.3077e+00,
           3.6272e-01, -6.7978e-02,  5.0646e-01, -9.8139e-02, -3.5561e-01,
           2.1193e-01, -3.2112e-01, -1.0640e+00,  1.1522e+00,  1.6295e-01,
          -9.1810e-02,  2.5271e-01,  1.3121e+00,  4.1859e-01,  4.8650e-01,
          -7.6771e-01, -2.1098e+00, -1.0039e+00,  7.3990e-01,  5.9568e-01,
           4.0183e-02, -4.6189e-01,  2.3483e-01, -2.4623e-01,  3.7889e-01,
           2.6964e-01,  1.2817e+00,  4.1513e-01, -5.9775e-01, -4.8990e-01,
           8.4952e-01, -5.1729e-01,  6.6830e-02, -1.3550e+00, -1.0791e+00,
          -1.0917e+00,  9.8484e-01,  1.3449e+00, -7.5181e-01, -1.1791e+00,
           2.4495e-02,  1.1747e+00, -7.9226e-01, -6.6300e-01,  1.6514e+00,
          -2.7893e-01, -7.1267e-02, -7.5119e-01, -7.7354e-01, -1.3604e+00,
           1.8475e+00, -3.6633e-01, -1.8304e+00, -1.6212e-01, -7.1406e-01,
           1.2788e-01,  2.2721e-01, -5.4311e-01,  8.8030e-01,  1.7222e+00,
          -1.0248e+00,  1.3961e+00,  1.1170e+00,  3.9953e-01, -9.8437e-01,
          -1.7904e-01,  1.2133e+00, -7.0255e-01, -6.3078e-01, -9.7471e-01,
          -8.8646e-01, -8.1072e-01,  6.3741e-02, -1.3988e+00, -1.2601e+00,
          -8.0972e-01,  5.9779e-01, -1.0549e+00, -2.1870e-01,  6.0083e-01,
          -8.2333e-01, -7.1244e-01, -9.6279e-01,  2.2684e-01,  1.6519e+00,
          -7.8967e-01,  3.2700e-03, -1.2214e-01,  7.6578e-01,  7.8383e-01,
           2.3100e-01, -4.6710e-01, -9.5071e-01, -2.3871e-01,  4.7464e-01,
          -4.2566e-01,  4.2463e-01,  7.3128e-01, -6.6968e-01, -7.4725e-01,
          -4.5471e-02, -1.0377e+00, -4.5282e-01,  2.2475e-01,  4.4936e-01,
           1.6603e-02,  1.1868e+00, -1.0460e+00, -4.1270e-01,  7.6347e-01,
           8.3294e-01,  1.0371e+00, -2.3788e+00,  7.7344e-01, -1.3298e+00,
           7.7145e-01,  2.1184e-01,  8.4736e-02, -2.5630e-01, -1.6933e+00,
          -2.1195e-01, -7.0515e-01,  2.6184e-01,  2.4401e-01, -3.5016e-01,
           6.6296e-02,  1.6248e-01, -1.1705e+00,  9.0456e-01, -5.7516e-01,
          -1.6785e-01, -1.0523e+00,  2.2173e-01, -8.0060e-01, -1.7748e+00,
          -7.7816e-01,  5.5475e-01, -1.5879e+00, -1.6239e+00,  1.4231e+00,
          -2.2266e-01, -5.1857e-01, -3.2148e-01, -5.2744e-01,  7.0523e-01,
          -4.0545e-01,  8.5339e-01, -1.5080e+00,  4.4426e-01, -1.6646e+00,
          -3.4879e-01, -9.3151e-01,  8.0067e-01, -4.2698e-01, -5.5647e-01,
          -8.6267e-01, -1.2337e+00, -2.0479e-01,  1.0907e-01, -8.4450e-01,
          -5.3496e-01,  2.7581e-01,  6.2063e-01, -4.7629e-01, -2.6022e+00,
          -2.4290e-01,  6.0470e-02,  1.1100e+00, -1.2858e+00, -7.1326e-01,
          -1.4877e-01, -1.5588e-02, -9.9140e-01, -7.5324e-01,  7.5144e-01,
           7.7801e-01,  7.5399e-01, -1.7745e+00, -2.5122e-01, -2.3623e-01,
          -3.1963e-02, -4.7449e-01, -7.9353e-01,  4.5337e-01,  1.5089e+00,
           4.5686e-02,  2.5291e-02,  7.4581e-01,  2.9301e-01,  7.9629e-01,
           5.3044e-01,  9.3968e-01,  1.5003e+00, -1.2838e+00,  1.5103e-01,
           1.3576e+00, -6.7148e-01,  6.4886e-01, -9.4079e-01, -1.2939e+00,
           6.4148e-01,  2.5995e+00,  8.4984e-01, -6.1686e-01,  1.6367e-01,
          -1.2862e-01,  2.7294e-02,  2.1019e-01,  3.4684e-01,  1.9525e-02,
          -5.4434e-01,  4.0012e-01, -7.2347e-03,  2.5163e+00, -1.1303e-02,
           3.9374e-02,  1.0235e-01,  1.2236e+00, -2.2061e-02, -6.1891e-01,
           5.2946e-02,  3.8842e-01,  4.1306e-02, -1.0189e+00, -1.0559e+00,
          -1.3311e+00, -1.1811e+00, -3.4012e-01, -1.8566e+00, -5.5514e-01,
           3.1717e-01, -5.1152e-01,  1.6059e-01,  6.5664e-01,  1.3852e-01,
          -1.0180e+00,  8.3119e-01, -6.3863e-01,  5.4885e-01,  7.1406e-01,
           3.3000e-01,  1.2693e-02,  3.2520e-01,  1.3472e+00,  1.0033e-02,
           1.1835e+00, -5.4702e-01, -1.7829e+00, -4.5449e-01, -7.3062e-01,
          -6.0173e-01,  1.6569e+00,  9.7329e-01,  9.0660e-02,  1.6251e+00,
           3.6985e-01,  6.6310e-01, -5.5031e-01,  5.7623e-01,  1.2680e+00,
           5.6095e-01, -5.0126e-01, -1.9631e-01,  3.8194e-01,  1.4074e+00,
          -1.1727e+00,  8.3819e-01,  5.0796e-01,  1.6244e+00, -2.4659e-01,
           1.4258e+00, -1.4496e+00, -6.3633e-01,  1.2703e-01, -2.0296e-01,
           1.3266e-01, -5.8688e-01, -6.6825e-01, -5.0643e-01, -1.4677e-01,
           9.3012e-01, -2.5461e-01,  9.1466e-01,  1.3532e+00, -5.4484e-01,
          -2.4148e+00,  4.0330e-01, -9.3913e-01,  1.1484e+00,  1.1952e+00,
          -6.0754e-01,  2.2694e+00,  2.3859e+00,  1.1664e+00, -7.0026e-01,
          -6.0696e-01,  1.2882e+00,  3.5842e-01, -4.1862e-01,  8.3419e-01,
           8.6348e-01,  9.0432e-01, -6.8639e-02,  1.2707e-01,  2.7780e-02,
          -5.4741e-01, -7.3692e-01,  1.8642e+00,  1.4173e-01, -5.0585e-01,
          -6.7568e-02, -4.0126e-02,  1.4899e-01,  1.4510e-02,  1.1999e+00,
           2.8070e-01,  7.0356e-01,  1.7350e+00,  1.1786e+00,  2.1565e-01,
           1.2649e+00, -5.5247e-02, -7.5159e-01,  2.7502e-01, -8.8020e-01,
          -8.3925e-01,  2.9522e-01,  2.4896e-01,  3.5320e-01, -3.1881e-01,
           4.0111e-01,  5.7892e-01, -4.7382e-01,  1.2211e+00,  2.2523e+00,
          -7.2818e-01, -4.9778e-01, -5.5391e-01,  4.1876e-01, -1.2979e+00,
          -1.4232e+00,  1.5282e+00, -8.1131e-01,  2.0355e-01,  3.6323e-01,
           6.1340e-01, -2.9293e-01,  8.7743e-01,  1.4219e+00, -1.1783e+00,
          -5.7384e-02,  1.9782e-01, -4.8981e-01,  2.2627e+00, -3.8925e-03,
          -3.2963e-01,  1.0174e+00, -8.8025e-01, -2.5614e-01,  2.9565e-01,
          -2.5018e-02, -7.6160e-01, -6.4492e-01,  1.1964e+00,  3.0099e-01,
           3.9882e-01,  2.0896e+00,  2.6635e-01, -5.6569e-01,  1.1958e+00,
           1.1226e+00, -2.4919e-01, -9.4099e-01,  1.2219e+00, -4.3676e-01,
           1.1060e-01, -1.8077e-01, -2.7820e-01,  9.5478e-01,  6.8013e-01,
           2.9735e-01,  6.3489e-01,  3.0169e-01,  1.0265e+00, -4.7719e-02,
          -1.6207e-01,  1.1254e+00,  1.5333e+00,  1.1331e+00, -4.1427e-01,
           4.5491e-01,  3.7236e-01, -7.2375e-01, -1.7287e+00,  4.6772e-01,
           6.1947e-01,  2.2063e-01, -6.0579e-01,  2.5648e-01,  2.3426e-01,
           1.6508e+00, -5.7952e-01, -5.5716e-02,  1.4434e-01,  4.6615e-01,
           2.2772e+00, -6.6289e-01, -3.9047e-01, -4.8998e-01,  8.7994e-01,
           2.2692e-02, -5.0443e-01,  2.5245e-01,  1.9355e-01, -1.5533e-01,
           2.8573e-02,  7.2306e-01]]], device='cuda:0')
#### Non constrained ####
the x from the 0 step: tensor([[[-9.8369e-02, -1.2588e+00, -4.0083e-01,  7.1023e-01, -1.7260e+00,
          -6.9739e-01,  4.2465e-01,  3.3887e-01, -1.2370e+00,  1.0292e-01,
          -9.2355e-01, -2.0646e-01,  1.5320e+00, -7.2098e-01,  2.5795e-01,
           3.2838e-01,  5.6032e-01, -7.6883e-01, -1.3811e+00,  6.8415e-01,
           5.7951e-01, -3.2894e-01,  7.7527e-04, -1.4715e+00,  2.2241e-01,
           1.8960e-01, -2.6239e-02, -1.1573e+00,  7.6907e-01, -4.9675e-01,
          -5.1632e-01,  4.2768e-01,  7.7610e-01, -1.2815e-01, -4.6923e-01,
           1.2469e+00, -7.2078e-01, -1.1556e+00,  2.2501e-02, -9.0010e-01,
          -7.3002e-02, -1.0166e-01,  1.6272e+00, -1.9627e-01,  1.1489e+00,
           1.8044e-01,  8.4395e-02,  1.7086e-01,  7.1007e-01,  9.8161e-01,
           5.6996e-01,  1.6703e-01, -1.4095e+00,  3.3906e-01, -3.2020e-01,
          -1.2261e-01,  2.4299e-03,  1.8288e-02, -4.5090e-01,  1.0472e-01,
          -2.6706e-01,  1.1488e+00, -2.5997e-01, -3.8027e-02, -4.0856e-01,
          -3.2677e-01,  3.5742e-01,  7.9243e-01,  7.2256e-01,  9.0619e-01,
          -1.3666e+00,  1.6813e+00, -1.1166e+00, -3.5064e-01,  2.7203e-01,
          -1.4429e+00, -4.5388e-01, -8.0395e-01, -4.2462e-01, -8.5272e-01,
          -1.8492e-01,  2.0460e-01, -1.1730e+00, -1.2089e+00, -1.6380e+00,
           1.0064e-01, -6.0615e-01, -2.9607e-01, -1.3502e+00,  1.5232e+00,
           2.4566e-01,  5.5703e-02, -2.6409e-01,  1.1281e-01,  4.0246e-01,
          -3.9996e-01, -2.5567e-01,  1.5014e-01, -4.1082e-02, -6.4010e-01,
           8.5321e-01, -3.7806e-01, -1.6045e+00,  1.7524e+00,  5.8299e-01,
           2.7727e-02, -2.0785e-01,  1.5594e+00, -4.3619e-01, -8.9970e-01,
          -7.7106e-01, -1.7304e+00, -8.4639e-01,  3.3091e-01,  3.4613e-01,
          -4.8292e-02, -2.2505e-01,  2.0162e-01, -6.2551e-01,  1.3060e+00,
          -4.9837e-01,  6.2854e-02,  1.1632e+00, -7.7019e-01,  1.4826e-01,
           1.8432e-01,  3.0021e-01, -6.4095e-02, -1.6557e+00, -9.9619e-01,
          -3.1413e-01,  7.7309e-02,  8.3861e-01, -1.6877e-01, -1.0294e+00,
           1.1851e-01,  1.6154e+00, -1.1800e+00, -1.2667e+00,  4.2718e-01,
          -4.3216e-01, -1.2712e-01, -1.5396e+00, -2.8580e-01, -1.3522e+00,
           1.7229e+00,  3.2809e-01, -1.8788e+00, -7.5116e-02, -2.6244e-01,
          -3.3837e-01,  1.4669e-01, -8.4763e-01,  3.8638e-01,  9.3247e-01,
          -1.0838e+00,  7.5826e-01,  1.3678e-01, -6.7635e-01, -1.3561e+00,
          -1.1816e-01,  1.2582e+00, -2.0518e+00, -1.3421e+00, -1.2128e+00,
          -1.2909e+00, -1.0689e+00, -1.1347e-01, -8.2007e-01, -1.2936e+00,
          -7.6808e-01,  6.8501e-01, -7.7899e-01,  7.8872e-01, -1.2488e-01,
          -1.0966e-01, -7.2105e-02, -3.3251e-01,  5.8241e-01,  2.0028e+00,
           8.4372e-02,  3.6247e-01, -1.2460e-01,  4.6943e-01,  3.5958e-01,
          -4.6346e-01, -3.8762e-01, -1.0352e+00, -1.0747e-01, -1.8936e-02,
          -6.5966e-02,  8.3977e-01,  4.8307e-01, -9.4335e-01, -5.9971e-01,
          -5.6525e-01,  5.1283e-02, -1.1911e+00,  5.6293e-01, -4.3194e-01,
          -6.8539e-01,  5.1338e-01, -2.7577e+00, -8.3106e-01, -1.1847e-02,
           5.9172e-01,  7.1597e-01, -2.7047e+00,  4.8316e-01, -6.5308e-01,
           5.9728e-01, -1.9102e-02,  3.5186e-01, -4.9055e-01, -1.1225e+00,
          -6.3729e-01, -1.2466e+00, -5.8916e-01, -1.2390e-01, -4.3412e-01,
           2.1889e-01,  2.1807e-02, -9.6880e-01,  5.9970e-01, -4.7881e-01,
           6.1608e-02, -1.2414e+00, -1.0823e+00, -4.7752e-01, -8.1135e-01,
          -4.6237e-01,  1.0890e+00, -1.2091e+00, -1.5394e+00,  1.0141e+00,
          -9.5753e-01, -4.4911e-01, -4.1729e-01, -3.9117e-01,  2.8118e-01,
          -3.0861e-01,  6.1166e-01, -1.5295e+00, -4.3775e-01, -7.1894e-01,
          -1.6280e+00, -8.4935e-01,  4.0125e-01, -1.0645e+00,  5.1290e-02,
          -5.0409e-01, -4.1793e-01, -2.9482e-01, -5.4630e-01, -1.0388e+00,
          -4.9692e-01, -3.0281e-01,  1.0358e-01, -7.1153e-01, -2.0219e+00,
          -1.4660e+00,  3.2251e-03, -3.1457e-01, -1.4374e+00, -7.6791e-01,
           3.0985e-01, -3.1339e-02, -1.0664e+00, -1.3509e+00,  8.2463e-01,
           6.4715e-01,  1.5842e-01, -2.3097e+00, -4.2878e-01, -4.4261e-01,
          -4.3147e-01, -1.3379e-01, -5.3104e-01,  2.0109e-01,  1.3229e+00,
           4.9649e-01,  2.9783e-01,  1.1055e+00, -7.1085e-01,  1.3956e+00,
           9.5957e-01,  1.2974e+00,  1.6448e+00, -1.1276e+00, -1.9712e-01,
           1.0277e+00,  9.5776e-01, -1.6927e-01, -4.2471e-01, -1.2086e+00,
           1.0305e+00,  1.7635e+00,  3.7781e-01, -8.0822e-01,  1.4172e+00,
           1.6106e-01,  3.0951e-01, -9.1487e-02,  2.4859e-01,  8.3132e-01,
          -7.4889e-01,  3.4412e-01,  8.6628e-01,  1.7518e+00,  3.5601e-01,
           4.4276e-01, -1.5844e-01,  8.7809e-01,  1.7328e-01,  1.4485e-01,
           8.2367e-01,  3.1974e-01,  2.4934e-01, -1.7314e-02, -9.7665e-01,
           1.6139e-01, -6.4356e-01,  2.0943e-01, -1.1707e+00,  2.0001e-01,
          -1.5364e-02, -5.7684e-01,  5.9018e-01,  4.1857e-01,  1.2569e+00,
          -2.8632e-01,  5.9494e-01, -1.0581e+00, -1.0514e-02,  2.5535e-01,
           6.8188e-01,  2.6234e-01,  1.5435e+00,  3.5512e-01, -1.3743e+00,
           1.2219e+00, -2.2324e-01, -1.4886e+00, -1.5257e+00, -2.8772e-01,
          -1.5178e-01,  1.4265e+00,  9.4844e-01,  1.9020e-01,  2.1656e+00,
           4.9358e-01,  7.2004e-01,  3.9877e-01,  7.5066e-01,  1.3654e+00,
           1.5422e+00, -3.4443e-01,  2.8606e-01,  3.4739e-02,  1.2593e+00,
          -1.1887e+00,  1.4472e+00,  1.5236e-01,  9.3755e-01,  4.6465e-01,
           6.0762e-01, -7.2020e-01, -1.0389e-01,  4.5505e-01, -2.9033e-01,
           1.0809e-01,  6.3048e-01, -4.8164e-01, -2.1022e-01, -5.0873e-01,
           2.8215e-01,  5.2812e-01,  7.8246e-01,  2.1204e+00, -4.8572e-01,
          -8.2638e-01,  9.6626e-01, -1.6981e+00,  1.0346e+00,  1.4434e+00,
          -1.3028e+00,  2.1603e+00,  2.1959e+00, -2.1621e-01, -2.7813e-01,
          -1.1869e-01,  1.1796e+00,  8.8143e-02,  9.9428e-01,  8.0380e-01,
           5.9527e-01,  4.0396e-01,  9.5308e-01,  1.0718e+00,  1.1756e-01,
          -1.9106e-01,  3.6710e-02,  2.0666e+00,  2.0117e-01,  2.9615e-01,
          -9.8857e-01, -3.6730e-01, -2.9757e-01, -1.9301e-01,  8.2507e-01,
           1.4219e+00,  9.4540e-01,  1.5371e+00,  1.4668e+00, -2.6964e-02,
           6.0370e-01,  6.9976e-02, -1.0299e+00, -4.8042e-01, -1.1185e+00,
          -1.2950e-01,  2.4148e-01,  1.0619e+00, -8.6286e-03, -1.8212e-02,
           6.5558e-01, -1.8394e-01, -1.3932e+00,  9.3575e-01,  4.8486e-01,
          -7.5699e-02, -3.7065e-02, -7.8328e-01,  9.4910e-01, -8.2523e-01,
          -9.6068e-01,  2.3464e+00, -3.0940e-01, -1.4366e-02,  4.2768e-01,
           6.0718e-02, -1.3503e-01,  7.5758e-01,  1.1131e+00, -1.5882e-01,
           1.1366e+00,  3.1747e-01,  4.5355e-02,  2.5137e+00,  9.7965e-02,
          -5.2368e-01,  2.8064e-01,  4.3883e-01, -1.0845e+00, -1.0097e-01,
           8.0640e-01, -6.8576e-02, -2.3002e-01,  6.9480e-01,  7.6101e-01,
           9.0638e-02,  1.5558e+00,  8.8744e-01, -4.1297e-01,  1.4918e+00,
           1.7195e+00,  1.1906e+00, -7.1120e-01,  1.6604e+00,  3.0810e-01,
          -8.4876e-01, -3.3300e-01, -2.1913e-01,  6.7907e-01,  5.0859e-01,
          -1.5067e-01, -2.9102e-01, -4.0683e-01,  1.1447e+00,  4.4592e-01,
          -8.3114e-01,  9.3937e-01,  2.1291e+00,  8.2871e-01, -5.9995e-02,
           6.4131e-01,  2.8055e-01, -4.8064e-01, -8.7089e-01,  5.2257e-01,
          -6.7681e-03,  1.1596e-01, -1.3179e+00,  1.0481e+00,  4.9271e-01,
           4.3657e-01, -1.2514e+00,  1.8542e-01,  2.6087e-02,  6.5609e-01,
           1.7118e+00,  5.2250e-01, -2.7297e-01,  2.4870e-01,  1.1751e-01,
          -4.5512e-01, -9.4961e-01, -6.6230e-01, -5.6117e-01,  5.9098e-01,
           1.1470e+00,  3.0555e-01]]], device='cuda:0')
incremental state:

# appears that the incremental state is changing the probabilities

[[-0.2688,  0.7609, -0.5229,  ...,  0.2351, -0.5366,  0.5300],
          [ 0.5314,  0.7448, -0.2648,  ...,  0.9351,  0.7723,  0.3032],
          [-0.2551, -0.2195, -0.3718,  ...,  0.1388,  0.1997, -0.2536]],

         [[-0.0226, -0.3284,  0.2200,  ..., -0.1178, -0.2866, -0.2465],
          [ 1.0572, -0.0626, -0.4491,  ..., -0.3095, -0.2996,  0.0055],
          [-0.0393,  0.0228, -0.1636,  ...,  0.2238, -0.0155, -0.0328]],

         [[-0.2154, -0.5580, -0.2680,  ...,  0.8886, -0.2217, -0.5730],
          [-0.0383,  0.3327,  0.5486,  ..., -0.5456,  0.6275, -0.3820],
          [-0.0044,  0.1666,  0.0907,  ...,  0.0377, -0.0447,  0.2532]]]],
       device='cuda:0'), 'prev_key_padding_mask': tensor([[0., 0., 0.]], device='cuda:0')}}]



"""