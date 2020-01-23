# -*- coding:utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/22 23:17
# @Author  : jxliu
# @Email   : jxliu.nlper@gmail.com
# @File    : evaluate.py

import sys
import codecs
from optparse import OptionParser
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

smoothing_function = SmoothingFunction().method4


def parse_opts():
  op = OptionParser()
  op.add_option(
    '-g', '--gold', dest='gold', type='str', help='path of gold')
  op.add_option(
    '-t', '--test', dest='test', type='str', help='path of predict')
  argv = [] if not hasattr(sys.modules['__main__'], '__file__') else sys.argv[
                                                                     1:]
  (opts, args) = op.parse_args(argv)
  if not opts.gold or not opts.test:
    op.print_help()
    exit()
  if opts.test:
    opts.train = False
  return opts


def readlines(path):
  file = codecs.open(path, 'r', encoding='utf-8')
  lines = []
  for line in file.readlines():
    line = line.strip()
    if not line:
      line = '#'
    lines.append(line)
  return lines


class Metrics(object):
  def __init__(self):
      pass

  @staticmethod
  def bleu_score(references, candidates):
    """Calculate BLEU score.
    Args:
      references: list(str), gold labels
      candidates: list(str), predict labels
    """
    bleu1_list = []
    bleu2_list = []
    bleu3_list = []
    bleu4_list = []
    for ref, cand in zip(references, candidates):
      ref_list = [list(ref)]
      cand_list = list(cand)
      bleu1 = sentence_bleu(
        ref_list, cand_list, weights=(1, 0, 0, 0),
        smoothing_function=smoothing_function)
      bleu2 = sentence_bleu(
        ref_list, cand_list, weights=(0.5, 0.5, 0, 0),
        smoothing_function=smoothing_function)
      bleu3 = sentence_bleu(
        ref_list, cand_list, weights=(0.33, 0.33, 0.33, 0),
        smoothing_function=smoothing_function)
      bleu4 = sentence_bleu(
        ref_list, cand_list, weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing_function)
      bleu1_list.append(bleu1)
      bleu2_list.append(bleu2)
      bleu3_list.append(bleu3)
      bleu4_list.append(bleu4)

    bleu1_average = sum(bleu1_list) / len(bleu1_list)
    bleu2_average = sum(bleu2_list) / len(bleu2_list)
    bleu3_average = sum(bleu3_list) / len(bleu3_list)
    bleu4_average = sum(bleu4_list) / len(bleu4_list)

    print("average bleus: bleu1: %.3f, bleu2: %.3f, bleu4: %.3f" % (
      bleu1_average, bleu2_average, bleu4_average))
    return (bleu1_average, bleu2_average, bleu4_average)

  @staticmethod
  def em_score(references, candidates):
    total_cnt = len(references)
    match_cnt = 0
    for ref, cand in zip(references, candidates):
      if ref == cand:
        match_cnt = match_cnt + 1

    em_score = match_cnt / (float)(total_cnt)
    print("em_score: %.3f, match_cnt: %d, total_cnt: %d" % (
      em_score, match_cnt, total_cnt))
    return em_score

  @staticmethod
  def rouge_score(references, candidates):
    """Calculate ROUGE score.
    """
    rouge = Rouge()

    rouge1_list = []
    rouge2_list = []
    rougel_list = []
    for ref, cand in zip(references, candidates):
      ref = ' '.join(list(ref))
      cand = ' '.join(list(cand))
      rouge_score = rouge.get_scores(cand, ref)
      rouge_1 = rouge_score[0]["rouge-1"]['f']
      rouge_2 = rouge_score[0]["rouge-2"]['f']
      rouge_l = rouge_score[0]["rouge-l"]['f']

      rouge1_list.append(rouge_1)
      rouge2_list.append(rouge_2)
      rougel_list.append(rouge_l)

    rouge1_average = sum(rouge1_list) / len(rouge1_list)
    rouge2_average = sum(rouge2_list) / len(rouge2_list)
    rougel_average = sum(rougel_list) / len(rougel_list)

    print("average rouges, rouge_1: %.3f, rouge_2: %.3f, rouge_l: %.3f" \
          % (rouge1_average, rouge2_average, rougel_average))
    return (rouge1_average, rouge2_average, rougel_average)


if __name__ == '__main__':
  opts = parse_opts()
  candidates = readlines(opts.gold)
  references = readlines(opts.test)

  assert len(candidates) == len(references)

  Metrics.bleu_score(references, candidates)
  Metrics.rouge_score(references, candidates)
  Metrics.em_score(references, candidates)
