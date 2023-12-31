#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: run_glue.py
@author: ImKe at 2022/3/20
@email: tuisaac163@gmail.com
@feature: based on Optimus run_glue.py
"""
from latent_classifier import AdaVAEforLatentClassification
import datetime, os, copy, math, time, collections, argparse, nltk, json, sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
sys.path.append('../')
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from src.logger import Logger
from src.adapters.vae import *
from src.utils import *
from torch.cuda import amp
from src.adapters.common import AdapterConfig
from src.data import DictDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW
from utils_glue import processors


parser = argparse.ArgumentParser()
scaler = amp.GradScaler()

# Default parameters are set based on single GPU training
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument('--iterations', type=int, default=10000 * 3)
parser.add_argument('--dataset', type=str, default='yelp', choices=['yelp', 'rejection_stratified', 'imdb', 'cola', 'sst-2', 'mrpc', 'wnli'],
                    help="Dataset to use for training")
parser.add_argument('--warmup', type=int, default=1000,
                    help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")

## mode options
parser.add_argument('--adapter_size', type=int, default=128,
                    help="Hidden size of GPT2 encoder/decoder adapter")
parser.add_argument('--latent_size', type=int, default=32,
                    help="Hidden size of latent code")
parser.add_argument('--encoder_n_layer', type=int, default=8,
                    help="attention layer number of GPT-2 encoder")
parser.add_argument('--decoder_n_layer', type=int, default=12,
                    help="attention layer number of GPT-2 decoder")
parser.add_argument('--label_size', type=int, default=2,
                    help="class number for classification")
# parser.add_argument('--label_emb_size', type=int, default=8,
#                     help="label embedding size")
parser.add_argument('--adapter_scalar', type=str, default="1.0",
                    help="adapter scalar")
parser.add_argument('--ffn_option', type=str, default="parallel_ffn",
                    choices=['sequential', 'parallel_attn', 'parallel_ffn', 'pfeiffer'],
                    help="adapter type option")
parser.add_argument('--attn_mode', type=str, default="none",
                    choices=['prefix', 'adapter', 'lora', 'none'],
                    help="attention transfer type")
parser.add_argument('--hidden_dropout_prob', type=float, default=0.1,
                    help="dropout rate for classifier logits")

## training paramters
parser.add_argument('--batch_sizes', nargs='+', type=int, default=[10],
                    help='batch size per GPU. Lists the schedule.')
parser.add_argument('--percentage_per_label', type=float, default=1.0)
parser.add_argument("--sample_per_label", type=int, default=-1,
                        help="Set this value, if you are using a subset of training dataset, and a fixed number of samples are specified.")
parser.add_argument('--eval_batch_size', type=int, default=100,
                    help='batch size per GPU. Lists the schedule.')
parser.add_argument('--seq-lens', nargs='+', type=int, default=[30],
                    help='seq length per sample. Lists the schedule.')
parser.add_argument('--max_length', type=int, default=25,
                    help='max length of every input sentence')
parser.add_argument('--label', type=int, default=0)
parser.add_argument('--n_samples', type=int, default=50)
parser.add_argument('--valid_run', type=int, default=1)

parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--out_dir', type=str, default='train_out')
parser.add_argument('--load_dir', type=str, default='')
parser.add_argument('--experiment', type=str, default=None)
parser.add_argument('--eval_output_dir', type=str, default='eval_out')
parser.add_argument('--restore_folder', type=str, default=None)
parser.add_argument('--adapter_init', type=str, default='bert', choices=['lora', 'bert', 'lisa', 'other'],
                    help="parameter initialization method for adapter layers.")
parser.add_argument('--latent_gen', type=str, default="latent_attn",
                    help="method for encoder to latent space, averaged_attn for average attention from "
                         "TransformerCVAE, linear for taken the first encoder token to a linear like Optimus",
                    choices=['latent_attn', 'averaged_attn', 'linear', 'mean_max_linear'])
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers')

# use GPU
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--no_gpu', action="store_true")

parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)

# loss weights
parser.add_argument('--beta_cls', default=1.00, type=float)
parser.add_argument('--beta_latent', default=0.50, type=float)
parser.add_argument('--beta_warmup', type=int, default=2000)


## trigger
parser.add_argument('--load', action="store_true")
parser.add_argument('--do_train', action="store_true")
parser.add_argument('--use_mean', action="store_true")
parser.add_argument('--feature_based', action="store_true", help="freeze backbone network")
parser.add_argument('--finetune_enc', action="store_true")
parser.add_argument('--attn_proj_vary', action="store_true")
parser.add_argument('--save_all', action="store_true", help="save full parameters of the model")


def tokenize(input_example, tokenizer, device, max_length):
    x_tokenized = tokenizer(input_example['text_a'], padding=True, truncation=True,
                                 max_length=max_length,
                                 return_tensors='pt')
    input_ids = x_tokenized['input_ids'].to(device)
    attention_mask = x_tokenized['attention_mask'].to(device)
    segment_ids = None

    if 'text_b' in input_example.keys():
        x_tokenized_b = tokenizer(input_example['text_b'], padding=True, truncation=True,
                                max_length=max_length,
                                return_tensors='pt')
        input_ids_b = x_tokenized_b['input_ids'].to(device)
        attention_mask_b = x_tokenized_b['attention_mask'].to(device)
        segment_ids = torch.zeros_like(input_ids, device=device)
        segment_ids_b = torch.ones_like(input_ids_b, device=device)

        input_ids = torch.cat([input_ids, input_ids_b], 1)
        attention_mask = torch.cat([attention_mask, attention_mask_b], 1)
        segment_ids = torch.cat([segment_ids, segment_ids_b], 1)

    return input_ids, attention_mask, segment_ids

def compute_loss(device, model, input_tokens, att_mask, seg_ids, labels):
    input_tokens = input_tokens.to(device)
    att_mask = att_mask.to(device)

    loss, representations, logits = model(input_ids=input_tokens, labels=labels,
                                          attention_mask=att_mask, token_type_ids=seg_ids)

    a, y_train = torch.max(logits, dim=1)
    train_acc = accuracy_score(labels.cpu(), y_train.cpu())
    return loss, train_acc


def train_step(device, model, optimizer, input_tokens, att_mask, seg_ids, labels):
    optimizer.zero_grad()
    loss, acc = compute_loss(device, model, input_tokens, att_mask, seg_ids, labels)

    loss = loss.mean()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # max_grad_norm=1.0
    scaler.step(optimizer)
    scaler.update()

    return loss, acc

def test_step(model, input_tokens, att_mask, seg_ids, y_hat, mcc=False):
    loss, representations, logits = model(input_ids=input_tokens, labels=y_hat,
                                          attention_mask=att_mask, token_type_ids=seg_ids)
    a, y = torch.max(logits, dim=1)
    ## y_true, y_pred
    test_acc = accuracy_score(y_hat.cpu(), y.cpu())
    test_recall = recall_score(y_hat.cpu(), y.cpu(), average='macro')
    test_precision = precision_score(y_hat.cpu(), y.cpu(), average='macro')
    test_f1 = f1_score(y_hat.cpu(), y.cpu(), average='macro')

    if mcc:
        tn, fp, fn, tp = confusion_matrix(y_hat.cpu().numpy(), y.cpu().numpy()).ravel()
        fz = tp * tn - fp * fn
        fm = (tp + fn) * (tp + fp) * (tn + fp) * (tn + fn)
        test_MCC = fz / pow(fm, 0.5)
    else:
        test_MCC = 0.

    return test_acc, test_recall, test_precision, test_f1, test_MCC, loss

def train(args):
    now = datetime.datetime.now()
    date = f"{now.month}.{now.day}"
    # GPU
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        # print('Setting GPUs {}'.format(args.device))
        # print('Using GPU devices {}'.format(devices))
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
    device = torch.device(args.gpu if gpu else "cpu")

    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    overall_loss, overall_acc, overall_prec, overall_recall, overall_f1 = [], [], [], [], []
    for ith_run in range(args.valid_run):
        seed_i = np.random.randint(0, 9999)
        # randomness
        np.random.seed(seed_i)
        torch.random.manual_seed(seed_i)
        if gpu:
            torch.cuda.manual_seed(seed_i)
            torch.cuda.manual_seed_all(seed_i)

        config = GPT2Config()
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        ada_config = AdapterConfig(hidden_size=768,
                                   adapter_size=args.adapter_size,
                                   adapter_act='relu',
                                   adapter_initializer_range=1e-2,
                                   latent_size=args.latent_size,
                                   class_num=args.label_size,
                                   encoder_n_layer=args.encoder_n_layer,
                                   decoder_n_layer=args.decoder_n_layer,
                                   init='other',
                                   adapter_scalar=args.adapter_scalar,
                                   ffn_option=args.ffn_option,
                                   attn_mode=args.attn_mode,
                                   attn_option='none',
                                   mid_dim=30,
                                   attn_bn=25,
                                   prefix_dropout=0.1,
                                   tune_enc=args.finetune_enc,
                                   tune_dec=False,
                                   latent_gen=args.latent_gen,
                                   dis_emb=128,
                                   add_z2adapters=False)  ## two-stage training, should employ plain GPT-2 decoder/encoder + adapters

        AdaVae_encoder = Encoder(config, ada_config)
        # AdaVae_average_attn = AverageSelfAttention(config.n_embd, ada_config)
        endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        model = AdaVAEforLatentClassification(args, config, AdaVae_encoder, use_mean=args.use_mean)
        ## load pre-trained weights
        init_para_frompretrained(model.encoder, gpt2_model.transformer, share_para=True)

        if args.load:
            prefix_load = "../src/out/"
            load_folder = os.path.join(prefix_load, args.load_dir)
            ## load ckpt
            print('Loading model weights...')
            state = torch.load(os.path.join(load_folder, 'model_best_val.pt'), map_location=device)  # , map_location='cpu' model_latest.pt
            if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
                state_copy = copy.copy(state)
                keys = state_copy.keys()
                for k in keys:
                    state[k.replace('module.', '')] = state.pop(k)
            ## load trained parameters
            if not args.save_all:
                model_dict = model.state_dict()
                additional_dict = {k: v for k, v in state.items() if k in model_dict}
                model_dict.update(additional_dict)
                model.load_state_dict(model_dict)  ## only loads adapters and latent connectors from ckpt
            else:
                model.load_state_dict(state)
        load = True if args.load else False
        final_folder = f"{args.dataset}_iter{args.iterations}_ft-{args.finetune_enc}_as-{args.adapter_size}_ls-{args.latent_size}_feature-{args.feature_based}_spl-{args.sample_per_label}_ppl-{args.percentage_per_label}_load{load}_{date}"

        save_folder = os.path.join(args.out_dir, final_folder)
        os.makedirs(save_folder, exist_ok=True)
        t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
        v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)
        logging_file = f"{args.dataset}_glue_{ith_run}.log"
        logging = Logger(os.path.join(save_folder, logging_file))

        model = model.to(device)

        model_params = num_params(model)
        logging.info(f'model params: {model_params}')

        # fix pre-trained parameters before certain iterations
        args.warmup = args.beta_warmup = int(args.iterations / 5)

        new_pars = ['classifier']
        if not args.feature_based:
            new_pars1 = ['c_z', 'attention_weights', 'mean', 'logvar', 'input_proj', 'attn_proj', 'Nu_fc1', 'Nu_fc2',
                    'lm_head_rep', 'lm_head']
            new_pars2 = ['label_embedding', 'latent_generator', 'linear', 'latent_classifier', 'latent_discriminator', 'conv1']
            new_pars.extend(new_pars1)
            new_pars.extend(new_pars2)

        for name, parameter in model.named_parameters():
            if not any([True if n in name else False for n in new_pars]):
                parameter.requires_grad = False

        encoder_unfreeze_modules = []
        if ada_config.attn_mode == "prefix":
            pass

        ## deal with encoder/decoder parameter settings
        if args.finetune_enc:
            for _, parameter in model.encoder.named_parameters():
                parameter.requires_grad = True
        else:
            for _, parameter in model.encoder.named_parameters():
                parameter.requires_grad = False
            model.encoder = unfreeze_GPT2_adapters(model.encoder, encoder_unfreeze_modules)

        for name, parameter in model.named_parameters():
            print((name, parameter.requires_grad))
        model_params_with_gradients = num_params(model)
        logging.info(f'model params with gradients:{model_params_with_gradients}')
        if args.finetune_enc:
            logging.info('Trainable parameters %d / %d= %.4f' % (model_params_with_gradients, model_params,
                                                                 model_params_with_gradients / model_params))
        else:
            logging.info('Additional parameters %d / %d = %.4f' % (model_params_with_gradients, model_params,
                                                                   model_params_with_gradients / (
                                                                           model_params - model_params_with_gradients)))

        logging.info('Setup data...')
        # Batch and sequence length schedule
        processor = processors[args.dataset]()
        if args.dataset == "yelp":
            prefix_data_path = "../data/yelp_polarity"
        elif args.dataset == "rejection_stratified":
            prefix_data_path = "../data/rejection_stratified"
        else:
            prefix_data_path = f"../glue_data/{args.dataset.upper()}"
        train_loader = DataLoader(
            DictDataset(processor.get_train_examples(prefix_data_path, args.percentage_per_label, args.sample_per_label)),
            batch_size=args.batch_sizes[0],
            pin_memory=True,
            drop_last=False,
            num_workers=args.workers,
            shuffle=True)
        val_loader = DataLoader(
            DictDataset(processor.get_dev_examples(prefix_data_path)),
            batch_size=args.batch_sizes[0],
            pin_memory=True,
            drop_last=False,
            num_workers=args.workers,
            shuffle=True)
        logging.info('Done.')

        def val_step(data_loader):
            max_val_batches = 1000
            ## use correlation flag
            mcc = True if args.dataset=="cola" else False
            model.eval()
            val_acc_list = []
            val_loss_list = []
            val_prec_list, val_recall_list, val_f1_list, val_mcc_list = [], [], [], []
            with tqdm(total=min(len(data_loader), max_val_batches), desc="Evaluating Model") as pbar:
                for i, val_data_dict in enumerate(val_loader):
                    input_tokens, att_mask, segment_ids = tokenize(val_data_dict, tokenizer, device, args.max_length)
                    y_hat = torch.as_tensor(val_data_dict['label'], dtype=torch.long).to(device)
                    with torch.no_grad():
                        ## loss, test_acc, test_recall, test_precision, test_f1
                        val_acc, val_recall, val_precision, val_f1, val_mcc, val_loss = test_step(model, input_tokens, att_mask, segment_ids, y_hat, mcc)
                        val_acc_list.append(val_acc.item())
                        val_loss_list.append(val_loss.item())
                        val_prec_list.append(val_precision.item())
                        val_recall_list.append(val_recall.item())
                        val_f1_list.append(val_f1.item())
                        val_mcc_list.append(val_mcc)
                    if i > max_val_batches:
                        break
                    pbar.update(1)
            val_loss = np.mean(val_loss_list)
            val_acc = np.mean(val_acc_list)
            val_prec = np.mean(val_prec_list)
            val_recall = np.mean(val_recall_list)
            val_f1 = np.mean(val_f1_list)
            val_mcc = np.mean(val_mcc_list)

            # with open(os.path.join(save_folder, "valid.txt"), "a") as f:
            #     f.write("iter{}\tloss: {:.4f}\tacc: {:.4f}\n".format(num_iters, val_loss, val_acc))
            if args.do_train:
                v_writer.add_scalar('val_loss', val_loss, num_iters)
                v_writer.add_scalar('val_acc', val_acc, num_iters)
                v_writer.add_scalar('val_precision', val_prec, num_iters)
                v_writer.add_scalar('val_recall', val_recall, num_iters)
                v_writer.add_scalar('val_f1', val_f1, num_iters)
            logging.info('val loss      : %.4f' % val_loss)
            logging.info('val acc       : %.4f' % val_acc)
            logging.info('val precision : %.4f' % val_prec)
            logging.info('val recall    : %.4f' % val_recall)
            logging.info('val f1        : %.4f' % val_f1)
            logging.info('val mcc       : %.4f' % val_mcc)
            model.train()

            return val_acc, val_loss, val_prec, val_recall, val_f1, val_mcc


        if args.do_train:
            # Batch and sequence length schedule
            assert len(args.batch_sizes) == len(args.seq_lens)
            batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens)))
            assert len(batch_schedule) <= 2, 'Currently not supporting multiple schedule'
            args.switch_time = 0
            cur_b_schedule = len(batch_schedule) - 1 if args.switch_time == 0 else 0
            logging.info('Batch schedule')
            logging.info(batch_schedule)
            logging.info('Wrapping models and optimizers...')
            # Apply linear scaling rule to increase batch size for short sequence training.
            lr_schedule = switch_schedule(linear_schedule(args), batch_schedule[cur_b_schedule][0] / batch_schedule[-1][0],
                                          int(args.iterations * args.switch_time))
            optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

            logging.info("Begin training iterations")
            logging.info("Total iteration: %d" % args.iterations)
            e = 0  # number of epoch
            num_iters = 0
            optimizer.zero_grad()

            best_acc, best_loss, best_prec, best_recall, best_f1, best_mcc = 0., 0., 0., 0., 0., 0.
            while num_iters < args.iterations:
                # Run epoch
                st = time.time()

                # Training
                print('Training loop. Batches:', len(train_loader))
                logging.info('\n----------------------------------------------------------------------')
                logging.info("Training loop.       Batches: %d" % len(train_loader))

                with tqdm(total=len(train_loader)) as pbar:
                    for i, data_dict in enumerate(train_loader):
                        input_tokens, att_mask, segment_ids = tokenize(data_dict, tokenizer, device, args.max_length)
                        labels = torch.as_tensor(data_dict['label'], dtype=torch.long).to(device)

                        if args.warmup != -1:
                            scheduler.step()

                        loss, acc = train_step(device, model, optimizer, input_tokens, att_mask, segment_ids, labels)

                        lr = scheduler.get_last_lr()[0]

                        loss = loss.item()

                        # Log to Tensorboard
                        t_writer.add_scalar('loss', loss, num_iters)
                        t_writer.add_scalar('acc', acc, num_iters)
                        t_writer.add_scalar('lr', lr, num_iters)
                        t_writer.add_scalar('iter_time', time.time() - st, num_iters)

                        st = time.time()
                        end = num_iters >= args.iterations

                        if end:
                            break
                        num_iters += 1
                        pbar.update(1)

                        log_var = int(args.iterations / 30)
                        if num_iters % log_var == 0:
                            logging.info("test set")
                            logging.info("validation set")
                            val_acc, val_loss, val_prec, val_recall, val_f1, val_mcc = val_step(val_loader)
                            if val_acc > best_acc:
                                best_acc = val_acc
                                best_loss = val_loss
                                best_prec = val_prec
                                best_recall = val_recall
                                best_f1 = val_f1
                                best_mcc = val_mcc
                                print('Saving model...')
                                logging.info("Iteration completed: %d, remained %d" % (num_iters, args.iterations - num_iters))
                                logging.info("Saving model...")
                                logging.info('\n------------------------------------------------------')

                                if args.adapter_size != 0:
                                    if args.save_all:
                                        save_orderdict = model.state_dict()
                                    else:
                                        save_orderdict = collections.OrderedDict()
                                        for name, parameter in model.named_parameters():
                                            if parameter.requires_grad:
                                                save_orderdict[name] = parameter
                                    torch.save(save_orderdict, os.path.join(save_folder, 'model_best_val.pt'))

                if not end:
                    e += 1
                    logging.info("Training loop. The ith epoch completed: %d" % e)

            if args.save_all:
                save_orderdict = model.state_dict()
            else:
                save_orderdict = collections.OrderedDict()
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        save_orderdict[name] = parameter
            torch.save(save_orderdict, os.path.join(save_folder, 'model_latest.pt'))

            logging.info('best loss      : %.4f' % best_loss)
            logging.info('best acc       : %.4f' % best_acc)
            logging.info('best precision : %.4f' % best_prec)
            logging.info('best recall    : %.4f' % best_recall)
            logging.info('best f1        : %.4f' % best_f1)
            logging.info('best mcc       : %.4f' % best_mcc)
            logging.info('Training complete.')

            overall_loss.append(best_loss)
            overall_acc.append(best_acc)
            overall_prec.append(best_prec)
            overall_recall.append(best_recall)
            overall_f1.append(best_f1)

        else:
            val_acc, val_loss, val_prec, val_recall, val_f1, val_mcc = val_step(val_loader)
            logging.info('test loss      : %.4f' % val_loss)
            logging.info('test acc       : %.4f' % val_acc)
            logging.info('test precision : %.4f' % val_prec)
            logging.info('test recall    : %.4f' % val_recall)
            logging.info('test f1        : %.4f' % val_f1)
            logging.info('test mcc       : %.4f' % val_mcc)

    logging.info('\n------------------------------------------------------')
    logging.info('Avg. loss      : %.4f' % np.mean(overall_loss))
    logging.info('Avg. acc       : %.4f' % np.mean(overall_acc))
    logging.info('Avg. precision : %.4f' % np.mean(overall_prec))
    logging.info('Avg. recall    : %.4f' % np.mean(overall_recall))
    logging.info('Avg. f1        : %.4f' % np.mean(overall_f1))

if __name__=="__main__":
    args = parser.parse_args()
    # args = parser.parse_args('--batch_sizes 100 --do_train --max_length 50 --dataset cola --iterations 200 --adapter_size 128 --percentage_per_label 1.0 --sample_per_label 50 --latent_size 768'.split())
    train(args)