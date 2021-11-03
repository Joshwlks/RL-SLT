Here I wil list all theplaces you have been working on inorder to enable you to quickly jump back in where you have left off

## Important Files
 
sequence_generator.py (def _generate --> where the encoder and decoder are being called)
models/transformer/transformer_decoder.py (def extract_features_scriptable --> where the decoder layers are being called.)
modules/transformer_layer.py (def forward --> The decoder layers are defined here)
tasks/fairseq_task.py (def inference step --> calls the generator and is how to pass previous tokens to the decoder)
utils.py
RL/interactive.py
data/dictionary.py