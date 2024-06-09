import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from transformers import AutoModelWithLMHead, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange
import torch
from pathlib import Path
import time
import json
import socket
import unicodedata
import subprocess
import threading
import traceback
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from torch.utils.tensorboard import SummaryWriter


chat_history_ids = None
last_activity_time = time.time()
counter = 0

tokenizer1 = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
tokenizer2 = AutoTokenizer.from_pretrained('lighteternal/gpt2-finetuned-greek')
models = {}
def find_models():
    script_directory = Path(__file__).parent
    root_path = Path(script_directory)
    folders = []
    for path in root_path.rglob('*'):
        if path.is_dir() and "cached" not in path.parts:
            folders.append(path)

    for folder in folders:
            try:
                model = AutoModelWithLMHead.from_pretrained(str(folder))
                model_name = folder.name
                models[model_name] = model
                print(f"Loaded model from: {folder}")
            except Exception as e:
                print(f"Could not load model from {folder}: {e}")


            
class UserSession:
    def __init__(self, client_socket, addr):
    
        self.client_socket = client_socket
        self.addr = addr
    def process_input_memory(self, user_input, model1):
        global chat_history_ids, last_activity_time, counter, tokenizer1, tokenizer2, models
        current_time = time.time()
        if current_time - last_activity_time > 30 or counter > 6:
            chat_history_ids = None
            print("Chat history reset due to inactivity.")

        if chat_history_ids is None:
            print("no history")

        new_user_input_ids = tokenizer1.encode(user_input + tokenizer1.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

        chat_history_ids = model1.generate(
            bot_input_ids, max_length=200,
            pad_token_id=tokenizer1.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        last_activity_time = time.time()
        counter += 1
        print("RickBot: {}".format(tokenizer1.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        return tokenizer1.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    def process_input_no_memory(self, user_input, model3):
        bot_input_ids = tokenizer1.encode(user_input + tokenizer1.eos_token, return_tensors='pt')
        chat_history_ids = model3.generate(
            bot_input_ids, max_length=100,
            pad_token_id=tokenizer1.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=10,
            top_p=0.7,
            temperature=0.8
        )
        print("Predict: {}".format(tokenizer1.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        return tokenizer1.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    def process_input_greek(self, user_input, model2):
        bot_input_ids = tokenizer2.encode(user_input + tokenizer2.eos_token, return_tensors='pt')
        chat_history_ids = model2.generate(
            bot_input_ids, max_length=15,
            pad_token_id=tokenizer2.eos_token_id,
            no_repeat_ngram_size=8,
            do_sample=True,
            top_k=150,
            top_p=0.7,
            temperature=1.0
        )
        print("Predict: {}".format(tokenizer2.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        return tokenizer2.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


    def convertToJson(selfself, model_name):

        # Load the data from a CSV file
        file_path = model_name  # Replace with your CSV file path
        data = pd.read_csv(file_path)

        # Shuffle the data and split into train and validation sets
        train_data, validate_data = train_test_split(data, test_size=0.2, random_state=42)

        # Function to convert DataFrame to JSON format (list of lists)
        def convert_to_json_format(df):
            return df.apply(lambda row: [row['user'], row['response']], axis=1).tolist()

        # Convert both subsets to the desired JSON format
        json_train_data = convert_to_json_format(train_data)
        json_validate_data = convert_to_json_format(validate_data)

        base_name = os.path.splitext(model_name)[0]
        model_name1 = base_name + 'train.json'
        model_name2 = base_name + 'validate.json'

        # Save the train data to a JSON file
        with open(model_name1, 'w', encoding='utf-8') as f:
            json.dump(json_train_data, f, ensure_ascii=False, indent=2)

        # Save the validate data to a JSON file
        with open(model_name2, 'w', encoding='utf-8') as f:
            json.dump(json_validate_data, f, ensure_ascii=False, indent=2)

        print("Data has been successfully saved to train.json and validate.json")




    def decode(self, model_name, prompt):
        import json
        import re
        import pandas as pd
        import unicodedata
        import os
        

        def parse_obj(obj):
            if isinstance(obj, str):
                return obj.encode('latin_1').decode('utf-8')

            if isinstance(obj, list):
                return [parse_obj(o) for o in obj]

            if isinstance(obj, dict):
                return {key: parse_obj(item) for key, item in obj.items()}

            return obj

        def is_number(s):
            try:
                float(s)  # Attempt to convert to float
                return True
            except ValueError:
                return False

        def extract_substring(s):
            # Take the first 10 characters of the string
            substr = s[:10]

            # Check if the last character is not a space and the substring is not the whole string
            if substr[-1] != ' ' and len(s) > 10:
                # Find the next space in the string after the 10th character
                next_space = s.find(' ', 10)

                # If there is a next space, extend the substring to the next space
                # If there isn't a next space, take the whole string
                substr = s[:next_space] if next_space != -1 else s

            return substr

        def is_specific_character(s):
            return s in [':', '!']

        def extract_content(messagesPrev, sender_name1, sender_name2):
            messages = messagesPrev["messages"]
            contents1 = []
            contents2 = []
            for i in range(len(messages) - 1):
                if messages[i]['sender_name'] == sender_name1 and messages[i + 1]['sender_name'] == sender_name2:
                    if 'content' in messages[i] and 'content' in messages[i + 1] and len(messages[i]) < 40 and len(
                            messages[i + 1]['content']) < 25 and 'http' not in messages[i]['content'] and 'http' not in \
                            messages[i + 1]['content'] and 'profit bird' not in messages[i]['content'] and 'profit bird' not in messages[i + 1]['content']:
                        flag = False
                        rr = messages[i]['content']
                        rr = ''.join(c for c in unicodedata.normalize('NFD', rr) if unicodedata.category(c) != 'Mn')
                        rr = re.sub(r'[,"\'\']', '', rr)
                        rr = re.sub(r'[^\w\s,;.?!:Ά-ώ]+', '', rr)
                        reply = messages[i + 1]['content']
                        k = i + 2
                        n = 0
                        while (messages[k]['sender_name'] == sender_name2) and n < 2:
                            if 'content' in messages[k] and len(messages[k]['content']) < 25 and 'http' not in \
                                    messages[k]['content'] and 'profit bird' not in messages[k]['content'] and 'video chat' not in messages[k]['content']:
                                reply += " "
                                reply += messages[k]['content']
                            k += 1
                            n += 1
                        reply = re.sub(r'[,"\'\']', '', reply)
                        reply = re.sub(r'[^\w\s,;.?!:Ά-ώ]+', '', reply)
                        reply = ''.join(
                            c for c in unicodedata.normalize('NFD', reply) if unicodedata.category(c) != 'Mn')

                        if not reply or len(rr) < 3 or len(rr) > 40 or not rr or is_number(rr) or is_specific_character(
                                rr) or is_number(reply) or is_specific_character(reply):
                            flag = True
                        if flag == False:
                            contents1.append(rr)
                            contents2.append(reply)
            return contents1, contents2


        # Path to your JSON file
        json_file_path = model_name
        print('decoding file...')
        print(json_file_path)
        sc1=[]
        sc2=[]
        with open((json_file_path), 'r') as f:
            file_contents = f.read()

        # Assuming the JSON structure matches what parse_obj expects
        decoded_data = parse_obj(json.loads(file_contents))

        reversed_data = {}
        for key, value in decoded_data.items():
            if isinstance(value, bool):
                # Handle boolean values differently (e.g., keep them unchanged)
                reversed_data[key] = value
            else:
                # Reverse the value (assuming it's a sequence, like a string)
                reversed_data[key] = value[::-1]

        user1, user2 = prompt.split(":::")

        sender_name1 = user1
        sender_name2 = user2

        decoded_data = json.dumps(reversed_data)

        sender_contents1, sender_contents2 = extract_content(json.loads(decoded_data), sender_name1, sender_name2)
        sc1 += sender_contents1
        sc2 += sender_contents2

        paired_messages = zip(sc1, sc2)
        print("Decoded data has been written to file.")
        # Create a DataFrame from the paired messages
        df = pd.DataFrame(paired_messages, columns=['user', 'response'])

        # Extract the base name without the extension
        base_name = os.path.splitext(model_name)[0]

        # Define the new file name with the .csv extension
        model_name = base_name + '.csv'

        # Save the DataFrame to a CSV file
        df.to_csv(model_name, index=False, encoding='utf-8')
        return model_name







    def createModel(self, model_name, prompt):
        from transformers import AutoModelWithLMHead, AutoTokenizer
        import torch
        import os



        """
        Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
        GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
        using a masked language modeling (MLM) loss.
        """
        original_name = os.path.splitext(model_name)[0]

        if model_name.endswith(".json"):
            print('converting from json')
            model_name = self.decode(model_name, prompt)
            self.convertToJson(model_name)

            base_name = os.path.splitext(model_name)[0]
            model_name1 = base_name + 'train.json'
            model_name2 = base_name + 'validate.json'

        elif model_name.endswith(".csv"):
            self.convertToJson(model_name)
            base_name = os.path.splitext(model_name)[0]
            model_name1 = base_name + 'train.json'
            model_name2 = base_name + 'validate.json'


        if "greek" in model_name:
            tokenizer = AutoTokenizer.from_pretrained("lighteternal/gpt2-finetuned-greek")
            model = AutoModelWithLMHead.from_pretrained("lighteternal/gpt2-finetuned-greek")
            tokenizerString = 'lighteternal/gpt2-finetuned-greek'
        else:
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-small")
            tokenizerString = 'microsoft/DialoGPT-small'
        # Configs
        logger = logging.getLogger(__name__)



        MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
        MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

        # Args to allow for easy convertion of python script to notebook
        class Args():
            def __init__(self):
                self.output_dir = original_name
                self.model_type = 'gpt2'
                self.model_name_or_path = tokenizerString
                self.config_name = tokenizerString
                self.tokenizer_name = tokenizerString
                self.cache_dir = 'cached'
                self.block_size = 512
                self.do_train = True
                self.do_eval = True
                self.evaluate_during_training = False
                self.per_gpu_train_batch_size = 4
                self.per_gpu_eval_batch_size = 4
                self.gradient_accumulation_steps = 1
                self.learning_rate = 5e-5
                self.weight_decay = 0.0
                self.adam_epsilon = 1e-8
                self.max_grad_norm = 1.0
                self.num_train_epochs = 3
                self.max_steps = -1
                self.warmup_steps = 0
                self.logging_steps = 1000
                self.save_steps = 3500
                self.save_total_limit = None
                self.eval_all_checkpoints = False
                self.no_cuda = False
                self.overwrite_output_dir = True
                self.overwrite_cache = True
                self.should_continue = False
                self.seed = 42
                self.local_rank = -1
                self.fp16 = False
                self.fp16_opt_level = 'O1'

        args = Args()

        f_train = open(model_name1)
        train_data = json.load(f_train)
        f_train.close()
        # print(len(train_data))

        f_validate = open(model_name2)
        validate_data = json.load(f_validate)
        f_validate.close()
        # print(len(validate_data))

        validate_contexted = []

        for i in range(len(validate_data)):
            row = []
            row.append(validate_data[i][1])
            row.append(validate_data[i][0])
            validate_contexted.append(row)

        train_contexted = []
        train_data = train_data

        for i in range(len(train_data)):
            row = []
            row.append(train_data[i][1])
            row.append(train_data[i][0])
            train_contexted.append(row)

        columns = ['response', 'context']
        columns = columns + ['context/' + str(i) for i in range(0)]
        columns

        len(train_contexted)
        trn_df = pd.DataFrame.from_records(train_contexted, columns=columns)
        trn_df.head(5)

        len(validate_contexted)
        val_df = pd.DataFrame.from_records(validate_contexted, columns=columns)
        val_df.head(5)

        def construct_conv(row, tokenizer, eos=True):
            flatten = lambda l: [item for sublist in l for item in sublist]
            conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
            conv = flatten(conv)
            return conv

        class ConversationDataset(Dataset):
            def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):

                block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

                directory = args.cache_dir
                cached_features_file = os.path.join(
                    directory, args.model_type + "_cached_lm_" + str(block_size)
                )

                if os.path.exists(cached_features_file) and not args.overwrite_cache:
                    logger.info("Loading features from cached file %s", cached_features_file)
                    with open(cached_features_file, "rb") as handle:
                        self.examples = pickle.load(handle)
                else:
                    logger.info("Creating features from dataset file at %s", directory)

                    self.examples = []
                    for _, row in df.iterrows():
                        conv = construct_conv(row, tokenizer)
                        self.examples.append(conv)

                    logger.info("Saving features into cached file %s", cached_features_file)
                    with open(cached_features_file, "wb") as handle:
                        pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, item):
                return torch.tensor(self.examples[item], dtype=torch.long)

        # Cacheing and storing of data/checkpoints

        def load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=False):
            return ConversationDataset(tokenizer, args, df_val if evaluate else df_trn)

        def set_seed(args):
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if args.n_gpu > 0:
                torch.cuda.manual_seed_all(args.seed)

        def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
            ordering_and_checkpoint_path = []

            glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

            for path in glob_checkpoints:
                if use_mtime:
                    ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
                else:
                    regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
                    if regex_match and regex_match.groups():
                        ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

            checkpoints_sorted = sorted(ordering_and_checkpoint_path)
            checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
            return checkpoints_sorted

        def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
            if not args.save_total_limit:
                return
            if args.save_total_limit <= 0:
                return

            # Check if we should delete older checkpoint(s)
            checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
            if len(checkpoints_sorted) <= args.save_total_limit:
                return

            number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
            checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
            for checkpoint in checkpoints_to_be_deleted:
                logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
                shutil.rmtree(checkpoint)

        trn_df

        def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
            """ Train the model """
            if args.local_rank in [-1, 0]:
                tb_writer = SummaryWriter()

            args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

            def collate(examples: List[torch.Tensor]):
                if tokenizer._pad_token is None:
                    return pad_sequence(examples, batch_first=True)
                return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

            train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate,
                drop_last=True
            )

            if args.max_steps > 0:
                t_total = args.max_steps
                args.num_train_epochs = args.max_steps // (
                            len(train_dataloader) // args.gradient_accumulation_steps) + 1
            else:
                t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

            model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
            model.resize_token_embeddings(len(tokenizer))
            # add_special_tokens_(model, tokenizer)

            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            )

            # Check if saved optimizer or scheduler states exist
            if (
                    args.model_name_or_path
                    and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
                    and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
            ):
                # Load in optimizer and scheduler states
                optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
                scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

            # multi-gpu training (should be after apex fp16 initialization)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # Distributed training (should be after apex fp16 initialization)
            if args.local_rank != -1:
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
                )

            # Train!
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_dataset))
            logger.info("  Num Epochs = %d", args.num_train_epochs)
            logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
            logger.info(
                "  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
            )
            logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
            logger.info("  Total optimization steps = %d", t_total)

            global_step = 0
            epochs_trained = 0
            steps_trained_in_current_epoch = 0
            # Check if continuing training from a checkpoint
            if args.model_name_or_path and os.path.exists(args.model_name_or_path):
                try:
                    # set global_step to gobal_step of last saved checkpoint from model path
                    checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
                    global_step = int(checkpoint_suffix)
                    epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                    steps_trained_in_current_epoch = global_step % (
                                len(train_dataloader) // args.gradient_accumulation_steps)

                    logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                    logger.info("  Continuing training from epoch %d", epochs_trained)
                    logger.info("  Continuing training from global step %d", global_step)
                    logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
                except ValueError:
                    logger.info("  Starting fine-tuning.")

            tr_loss, logging_loss = 0.0, 0.0

            model.zero_grad()
            train_iterator = trange(
                epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
            )
            set_seed(args)  # Added here for reproducibility
            for _ in train_iterator:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
                for step, batch in enumerate(epoch_iterator):

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        continue

                    inputs, labels = (batch, batch)
                    if inputs.shape[1] > 1024: continue
                    inputs = inputs.to(args.device)
                    labels = labels.to(args.device)
                    model.train()
                    outputs = model(inputs, labels=labels)
                    loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1

                        if args.local_rank in [-1,
                                               0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                            # Log metrics
                            if (
                                    args.local_rank == -1 and args.evaluate_during_training
                            ):  # Only evaluate when single GPU otherwise metrics may not average well
                                results = evaluate(args, model, tokenizer)
                                for key, value in results.items():
                                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                            tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                            logging_loss = tr_loss

                        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                            checkpoint_prefix = "checkpoint"
                            # Save model checkpoint
                            output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                            os.makedirs(output_dir, exist_ok=True)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving model checkpoint to %s", output_dir)

                            _rotate_checkpoints(args, checkpoint_prefix)

                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)

                    if args.max_steps > 0 and global_step > args.max_steps:
                        epoch_iterator.close()
                        break
                if args.max_steps > 0 and global_step > args.max_steps:
                    train_iterator.close()
                    break

            if args.local_rank in [-1, 0]:
                tb_writer.close()

            return global_step, tr_loss / global_step

        # Evaluation of some model

        def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, df_trn, df_val, prefix="") -> Dict:
            # Loop to handle MNLI double evaluation (matched, mis-matched)
            eval_output_dir = args.output_dir

            eval_dataset = load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=True)
            os.makedirs(eval_output_dir, exist_ok=True)
            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

            # Note that DistributedSampler samples randomly

            def collate(examples: List[torch.Tensor]):
                if tokenizer._pad_token is None:
                    return pad_sequence(examples, batch_first=True)
                return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate, drop_last=True
            )

            # multi-gpu evaluate
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # Eval!
            logger.info("***** Running evaluation {} *****".format(prefix))
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)
            eval_loss = 0.0
            nb_eval_steps = 0
            model.eval()

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                inputs, labels = (batch, batch)
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                with torch.no_grad():
                    outputs = model(inputs, labels=labels)
                    lm_loss = outputs[0]
                    eval_loss += lm_loss.mean().item()
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            perplexity = torch.exp(torch.tensor(eval_loss))

            result = {"perplexity": perplexity}

            output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            return result

        def main2(df_trn, df_val):
            args = Args()

            if args.should_continue:
                sorted_checkpoints = _sorted_checkpoints(args)
                if len(sorted_checkpoints) == 0:
                    raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
                else:
                    args.model_name_or_path = sorted_checkpoints[-1]

            if (
                    os.path.exists(args.output_dir)
                    and os.listdir(args.output_dir)
                    and args.do_train
                    and not args.overwrite_output_dir
                    and not args.should_continue
            ):
                raise ValueError(
                    "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                        args.output_dir
                    )
                )

            # Setup CUDA, GPU & distributed training
            # device = torch.device("cuda")
            # args.n_gpu = torch.cuda.device_count()
            # args.device = device

            device = torch.device("cpu")
            args.n_gpu = 0
            args.device = device

            # Setup logging
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
            )
            logger.warning(
                "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank,
                device,
                args.n_gpu,
                bool(args.local_rank != -1),
                args.fp16,
            )

            # Set seed
            set_seed(args)

            config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
            model = AutoModelWithLMHead.from_pretrained(
                args.model_name_or_path,
                from_tf=False,
                config=config,
                cache_dir=args.cache_dir,
            )
            model.to(args.device)

            logger.info("Training/evaluation parameters %s", args)

            # Training
            if args.do_train:
                train_dataset = load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=False)

                global_step, tr_loss = train(args, train_dataset, model, tokenizer)
                logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

            # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
            if args.do_train:
                # Create output directory if needed
                os.makedirs(args.output_dir, exist_ok=True)

                logger.info("Saving model checkpoint to %s", args.output_dir)
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)

                # Good practice: save your training arguments together with the trained model
                torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

                # Load a trained model and vocabulary that you have fine-tuned
                model = AutoModelWithLMHead.from_pretrained(args.output_dir)
                tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
                model.to(args.device)

            # Evaluation
            results = {}
            if args.do_eval and args.local_rank in [-1, 0]:
                checkpoints = [args.output_dir]
                if args.eval_all_checkpoints:
                    checkpoints = list(
                        os.path.dirname(c) for c in
                        sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                    )
                    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
                logger.info("Evaluate the following checkpoints: %s", checkpoints)
                for checkpoint in checkpoints:
                    global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                    prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                    model = AutoModelWithLMHead.from_pretrained(checkpoint)
                    model.to(args.device)
                    result = evaluate(args, model, tokenizer, df_trn, df_val, prefix=prefix)
                    result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                    results.update(result)

            return results

        main2(trn_df, val_df)

    def find_models(self, models, keyword):
        matched_models = {}
        for model_name, model in models.items():
            if keyword in model_name:
                return model

    def remove_diacritics(self, text):
        # Normalize the text to separate the base characters from the diacritics
        normalized_text = unicodedata.normalize('NFD', text)
        # Filter out the diacritical marks
        text_without_diacritics = ''.join([char for char in normalized_text if not unicodedata.combining(char)])
        return text_without_diacritics


    def handle(self):
        print(f'Connected by {self.addr}')
        try:
            data = self.client_socket.recv(1024).decode('utf-8').strip()
            opid, user_name, model_name, prompt = data.split("\n")
            print(opid, user_name, model_name, prompt)
            if opid == "op1":
                if model_name == "greek" or "greek" in model_name:
                        modeluse = self.find_models(models, model_name)
                        prompt = self.remove_diacritics(prompt)
                        response = self.process_input_greek(prompt, modeluse)
                        self.client_socket.sendall((response + "\n").encode('utf-8'))
                elif model_name == "rick" or "mm" in model_name:
                    print("using rick 1")
                    modeluse = self.find_models(models, model_name)
                    response = self.process_input_memory(prompt, modeluse)
                    self.client_socket.sendall((response + "\n").encode('utf-8'))
                elif model_name == "basic":
                    print("using basic")
                    modeluse = AutoModelWithLMHead.from_pretrained('lighteternal/gpt2-finetuned-greek')
                    response = self.process_input_greek(prompt, modeluse)
                    self.client_socket.sendall((response + "\n").encode('utf-8'))
                else:
                    modeluse = self.find_models(models, model_name)
                    response = self.process_input_no_memory(prompt, modeluse)
                    self.client_socket.sendall((response + "\n").encode('utf-8'))
                

                
            elif opid == "op2":
                
                print("File received")
                self.createModel(model_name, prompt)
                find_models()
        except Exception as e:
            print(f"Error handling client {self.addr}: {e}")
            print(traceback.format_exc())

        finally:
            self.client_socket.close()

    
    

def start_server():
    find_models()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 8091))
    server_socket.listen(5)
    print('Server started and listening for connections...')
    
    while True:
        client_socket, addr = server_socket.accept()
        user_session = UserSession(client_socket, addr)
        client_thread = threading.Thread(target=user_session.handle)
        client_thread.start()

if __name__ == "__main__":
    start_server()
