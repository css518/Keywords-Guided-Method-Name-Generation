python train_and_eval.py --node_vocab_file vocabulary/node_5.vocab --edge_vocab_file vocabulary/edge.vocab --target_vocab_file vocabulary/output_5.vocab --copy_attention --model_name kgmngen --checkpoint_dir kgmngen --node_features_dropout 0.0 --embeddings_dropout 0.2 --batch_size 16 --seed 2020 --rnn_hidden_size 256 --rnn_hidden_dropout 0.0 --beam_width 10 --infer_source_file kgmngen_data/test/inputs.jsonl.gz --infer_predictions_file kgmngen_data/test/infer_method_names.json --infer_target_file kgmngen_data/test/summaries.jsonl.gz
