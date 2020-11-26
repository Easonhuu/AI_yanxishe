import argparse
import time
import os
from transformers import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
from transformers import glue_processors as processors

MODEL_CONFIG_CLASSES = list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in MODEL_CONFIG_CLASSES), (),)

# debug parameters
data_dir = "./data"
output_dir = "./output"
log_dir = "./log"

# model_type = "xlnet"
# model_specific = "xlnet-base-cased"
# model_name_or_path = "xlnet-base-cased"
# config_name = "./pretrained_model/xlnet-base/xlnet-base-cased-config.json"

model_type = "roberta"
model_specific = "roberta-base"
model_name_or_path = "./pretrained_model/roberta-base/roberta-base-pytorch_model.bin"
config_name = "./pretrained_model/roberta-base/config.json"

# model_type = "roberta"
# model_specific = "xlnet-large-cased"
# model_name_or_path = "./pretrained_model/roberta-large/roberta-large-pytorch_model.bin"
# config_name = "./pretrained_model/roberta-large/roberta-large-pytorch_model.bin"

tokenizer_name = model_specific

log_file = os.path.join(log_dir, model_specific + '_' + time.strftime("%Y-%m-%d_%H-%M", time.localtime()) + '.txt')

output_mode = "regression"
cache_dir = ""
do_train = True
do_eval = True
do_test = True
is_cv = True
do_lower_case = True
evaluate_during_training = False
max_seq_length = 128
learning_rate = 1e-6
epochs = 2
gpu_start = 0
n_gpu = 4
device_ids = []  # IDs of GPUs to be used
for i in range(n_gpu):
    device_ids.append(gpu_start+i)
per_gpu_batch_size = 8
batch_size = per_gpu_batch_size * n_gpu 
early_stop_scale = 0.005
early_stop_num = 10
kfold = 5
gradient_accumulation_steps = 1
seed = 24
weight_decay = 0.0
adam_epsilon = 1e-8
max_grad_norm = 1.0
warmup_steps = 0
local_rank = -1

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument(
    "--data_dir",
    default=data_dir,
    type=str,
    # required=True,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
)
parser.add_argument(
    "--output_dir",
    default=output_dir,
    type=str,
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--log_dir",
    default=log_dir,
    type=str,
    help="The log directory where the model results will be written.",
)
parser.add_argument(
    "--model_type",
    default=model_type,
    type=str,
    help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
)
parser.add_argument(
    "--model_specific", 
    default=model_specific, 
    type=str
)
parser.add_argument(
    "--model_name_or_path",
    default=model_name_or_path,
    type=str,
    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
)
parser.add_argument(
    "--config_name", 
    default=config_name, 
    type=str, 
    help="Pretrained config name or path if not the same as model_name",
)
parser.add_argument(
    "--tokenizer_name",
    default=tokenizer_name,
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--log_file", 
    default=log_file, 
    type=str, 
)
parser.add_argument(
    "--output_mode",
    default=output_mode,
    type=str,
    help="regression or classification",
)
parser.add_argument(
    "--cache_dir",
    default=cache_dir,
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--do_train", 
    default=do_train, 
    action="store_true", 
    help="Whether to run training."
)
parser.add_argument(
    "--do_eval", 
    default=do_eval, 
    action="store_true", 
    help="Whether to run eval on the dev set."
)
parser.add_argument(
    "--do_test", 
    default=do_test, 
    action="store_true", 
    help="Whether to run testing."
)
parser.add_argument(
    "--is_cv", 
    default=is_cv, 
    action="store_true", 
    help="Whether to run cross validation."
)
parser.add_argument(
    "--do_lower_case", 
    default=do_lower_case, 
    action="store_true", 
    help="Set this flag if you are using an uncased model.",
)
parser.add_argument(
    "--evaluate_during_training", 
    default=evaluate_during_training, 
    action="store_true", 
    help="Run evaluation during training at each logging step.",
)
parser.add_argument(
    "--max_seq_length",
    default=max_seq_length,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--learning_rate", 
    default=learning_rate, 
    type=float, 
    help="The initial learning rate for Adam."
)
parser.add_argument(
    "--epochs", 
    default=epochs, 
    type=int, 
    help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--gpu_start", 
    default=gpu_start, 
    type=int, 
)
parser.add_argument(
    "--n_gpu", 
    default=n_gpu, 
    type=int, 
)
parser.add_argument(
    "--device_ids", 
    default=device_ids, 
    type=list, 
)
parser.add_argument(
    "--per_gpu_batch_size", 
    default=per_gpu_batch_size, 
    type=int, 
    help="Batch size per GPU/CPU.",
)
parser.add_argument(
    "--batch_size", 
    default=batch_size, 
    type=int, 
    help="Batch size.",
)
parser.add_argument(
    "--early_stop_scale", 
    default=early_stop_scale, 
    type=float,
)
parser.add_argument(
    "--early_stop_num", 
    default=early_stop_num, 
    type=int,
)
parser.add_argument(
    "--kfold", 
    default=kfold, 
    type=int,
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=gradient_accumulation_steps,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--seed", 
    default=seed, 
    type=int, 
    help="random seed for initialization"
)
parser.add_argument(
    "--weight_decay", 
    default=weight_decay, 
    type=float, 
    help="Weight decay if we apply some."
)
parser.add_argument(
    "--adam_epsilon", 
    default=adam_epsilon, 
    type=float, 
    help="Epsilon for Adam optimizer."
)
parser.add_argument(
    "--max_grad_norm", 
    default=max_grad_norm, 
    type=float, 
    help="Max gradient norm."
)
parser.add_argument(
    "--warmup_steps", 
    default=warmup_steps, 
    type=int, 
    help="Linear warmup over warmup_steps."
)
parser.add_argument(
    "--local_rank", 
    type=int, 
    default=local_rank, 
    help="For distributed training: local_rank"
)

args = parser.parse_args()
