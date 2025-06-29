import logging
import argparse
import os
import random
import numpy as np

import torch
from tqdm import tqdm

from models.modeling import VisionTransformer, CONFIGS
#from utils.data_utils import get_loader #modified
from utils.c_data_utils import get_loader
from utils.dist_util import get_world_size

logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)

    # Load pretrained weights from a .bin file
    checkpoint = torch.load(args.pretrained_dir, map_location=args.device)
    model.load_state_dict(checkpoint)

    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Testing parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def test(args, model, test_loader):
    # Test the model
    logger.info("***** Running Testing *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_labels = [], []

    epoch_iterator = tqdm(test_loader, 
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}", 
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])

    loss_fct = torch.nn.CrossEntropyLoss()

    test_losses = 0.0
    num_batches = 0
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0] # (batch_size, num_classes)

            test_loss = loss_fct(logits, y)
            test_losses += test_loss.item() # validation 때와는 달리 test 데이터셋에 대한 평균 손실만 구하면 되기 때문
            num_batches += 1

            average_loss = test_losses / num_batches
            epoch_iterator.set_description("Testing... (loss=%2.5f)" % average_loss)

            preds = torch.argmax(logits, dim=-1)

            #epoch_iterator.update()

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
            all_labels[0] = np.append(all_labels[0], y.detach().cpu().numpy(), axis=0)
        
    # Convert lists to numpy arrays
    # all_preds == all_preds[0] because of all_preds = [pred1_batch1, pred2_batch1, ..., predN_batchN]
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    accuracy = simple_accuracy(all_preds, all_labels)

    # Calculate average loss
    average_test_loss = test_losses / num_batches

    logger.info("\n")
    logger.info("Test Results")
    logger.info("Test Loss: %2.5f" % average_test_loss)
    logger.info("Test Accuracy: %2.5f" % accuracy)

    return accuracy

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True, help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet1k"], default="cifar10", help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"], default="ViT-B_16", help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz", help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str, help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=384, type=int, help="Resolution size")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Total batch size for training.")  
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=365, help="random seed for initialization")

    # Random Erasing
    parser.add_argument('--p', default=0.5, type=float, help='Random Erasing probability')
    parser.add_argument('--sh', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')

    parser.add_argument('--p2', default=0.5, type=float, help='random erasing in the frequency domain(REF) probability') # 0.3으로 되어 있었음(5/25)
    parser.add_argument('--grid_p', default=0.5, type=float, help='random erasing in the frequency domain(REF) grid probability')

    parser.add_argument('--p3', default=0.5, type=float, help='Hide And Seek probability') # 0.3
    parser.add_argument('--grid_ratio', default=0.04166, type=float, help='Hide And Seek grid ratio') # 0.125
    parser.add_argument('--patch_p', default=0.5, type=float, help='Hide And Seek patch') # 0.7

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1)))
    
    # Set seed
    set_seed(args)

    # Model Setup
    model = setup(args)

    # Prepare dataset
    _, _, test_loader, outloaders = get_loader(args)

    logger.info("Total number of batches in test_loader: %d", len(test_loader))

    # Testing on clean CIFAR Dataset
    logger.info("\Testing on CIFAR Dataset:")
    accuracy = test(args, model, test_loader)
    logger.info("Final Test Accuracy: \t%f" % accuracy)

    # Testing on CIFAR-C corruptions Dataset
    logger.info("\nTesting on Cifar-C corruptions:")
    acc_res = []
    for key, outloader in outloaders.items(): 
        accuracy = test(args, model, outloader)
        print("key, acuracy: ", key, accuracy)
        acc_res.append(accuracy)
        logger.info("%s Corruption Accuracy: \t%f" % (key, accuracy))
    
    logger.info("Mean Accuracy on CIFAR-C Dataset: \t%f" % np.mean(acc_res))


if __name__ == "__main__":
    main()
