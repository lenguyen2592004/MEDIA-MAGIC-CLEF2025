import argparse
import os
import sys
import torch
import json
from pathlib import Path
import ruamel_yaml as yaml

from models.model_vqa import MUMC_VQA
from models.vision.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn
from utils import init_distributed_mode, set_seed, is_main_process, Logger
from vqaEvaluate import compute_vqa_acc

@torch.no_grad()
def evaluation(model, data_loader, device, config):
    model.eval()
    result = []
    answer_list = [answer + config['eos'] for answer in data_loader.dataset.answer_list]

    for image, question, question_id in data_loader:
        image = image.to(device, non_blocking=True)
        topk_ids, topk_probs = model(image, question, answer_list, train=False, k=config['k_test'])

        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())
            _, pred = topk_prob.max(dim=0)
            result.append({"qid": ques_id, "answer": data_loader.dataset.answer_list[topk_id[pred]]})
    return result


def main(args, config):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    print("Creating vqa {} dataset".format(args.dataset_use))
    datasets = create_dataset(args.dataset_use, config)
    test_dataset = datasets[1]

    test_loader = create_loader([test_dataset], [None],
                                batch_size=[config['batch_size_test']],
                                num_workers=[4],
                                is_trains=[False],
                                collate_fns=[None])[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    print("Loading model")
    model = MUMC_VQA(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']

    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

    msg = model.load_state_dict(state_dict, strict=False)
    print("Loaded checkpoint from", args.checkpoint)
    print(msg)

    result = evaluation(model, test_loader, device, config)

    os.makedirs(args.result_dir, exist_ok=True)
    result_path = os.path.join(args.result_dir, 'vqa_inference_result.json')
    json.dump(result, open(result_path, 'w'))
    print("Saved result to", result_path)

    # Optional: compute accuracy
    if args.compute_acc:
        compute_vqa_acc(answer_list_path=config[args.dataset_use]['test_file'][0],
                        epoch=config['max_epoch'],
                        res_file_path=result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_use', default='rad', help='rad, pathvqa, slake')
    parser.add_argument('--checkpoint', required=True, help='path to model checkpoint')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--result_dir', default='./results/vqa')
    parser.add_argument('--compute_acc', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open('./configs/VQA.yaml', 'r'), Loader=yaml.Loader)

    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(filename=os.path.join(args.result_dir, "inference_log.txt"), stream=sys.stdout)

    print("config: ", config)
    print("args: ", args)
    main(args, config)
