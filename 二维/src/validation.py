import torch
from torch.autograd import Variable
import time
import sys

from utils import AverageMeter,calculate_accuracy

def val_epoch(epoch,data_loader,model,criterion,opt,logger):
    print('validation at epoch{}'.format(epoch) )
    model.eval()

    batch_time =AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i ,(inputs,labels) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        labels = list(map(int,labels))
        inputs = torch.unsqueeze(inputs,1)
        inputs = inputs.type(torch.FloatTensor)

        if not opt.no_cuda:
            labels = torch.LongTensor(labels).cuda(async=True)
        with torch.no_grad():
            inputs = Variable(inputs)
            labels = Variable(labels)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        losses.update(loss.data, inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
            epoch,
            i + 1,
            len(data_loader),
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            acc=accuracies))

    logger.log({'epoch': epoch, 'loss': losses.avg.item(), 'acc': accuracies.avg.item()})

    return losses.avg
