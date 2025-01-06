from option import args
import torch
import utility
from trainer import Trainer
from tester import Tester


if __name__ == '__main__':
    if args.test:
        t = Tester(args)
        t.generate_representation()
    else:
        checkpoint = utility.checkpoint(args)
        if checkpoint.ok:
            t = Trainer(args, checkpoint)
            t.train()

            checkpoint.done()