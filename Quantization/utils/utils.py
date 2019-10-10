import torch
from torchnet.engine import Engine
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import scipy.misc
import random
from io import StringIO, BytesIO


class TBLogger(object):
    r"""Logger to log data in tensorboard.

    Dowloaded from https://github.com/yunjey/pytorch-tutorial
    """

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def train(model, trainloader, testloader, loss_function, optimizer, lr_scheduler, maxepoch,
          device, tb_logfile='./tb_log', ckpt_file='./state.ckpt', ckpt_load=False, ckpt_extend=False):
    r"""Trains a given nn model and logs the progress.

    Training metrics, such as loss and accuracy, and histograms of
    weights and gradients can be visualised using TensorBoard.

    For checkpointing, the weights are stored after every epoch in the
    `ckpt_file` checkpoint file. To load from the previously stored checkpoint,
    set `ckpt_load` parameter to `True`.

    Arguments:
        model (:class:`torch.nn.Module`): nn model to train.
        trainloader (:class:`torch.utils.data.DataLoader`): dataloader to
            iterate through the training data and labels.
        testloader (:class:`torch.utils.data.DataLoader`): dataloader to
            iterate through the test data and labels.
        loss_function (:class:`torch.nn._Loss`): function to compute loss
        optimizer (:class:`torch.optim.Optimizer`): optimzer to update
            model's parameters.
        lr_scheduler (:class:`torch.optim._LRScheduler`): learning rate
            scheduler to adjust learning rate based on the number of epochs.
        maxepoch (int): maximum number of epochs.
        device (:class:`torch.device`): the device to run training on.
        tb_logfile (string): Tensorboard log file to log and
            visualize training metrics, such as loss, accuracy, weights,
            and gradient of weights, in Tensorboard.
        ckpt_file (string): name of the checkpoint file.
        ckpt_load (bool): flag to restart trainnig from the checkpoint state.
        ckpt_extend (bool): if starting from checkpoint state, flag to specify
            whether to extend training by `maxepoch` or run training until
            `maxepoch`.

    Returns:
        Module: trained model

    """
    start_epoch = 0
    if ckpt_load:
        start_epoch, model = load_checkpoint(model, ckpt_file)
    if ckpt_extend:
        maxepoch += start_epoch

    logger = TBLogger(tb_logfile)
    model.to(device)

    # Log initial weight values
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), 0)

    step = start_epoch * trainloader.__len__()
    for epoch in range(start_epoch, maxepoch):

        model.train()
        train_iter = tqdm(trainloader, desc='Epoch %03d' % (epoch + 1), leave=False)
        # Set learning rate
        if lr_scheduler:
            lr_scheduler.step(epoch + 1)
        for inputs, labels in train_iter:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()

            # Log training loss and accuracy
            __, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().mean()
            logger.scalar_summary('train/loss', loss, step)
            logger.scalar_summary('train/accuracy', accuracy, step)

            step += 1

        # Log weights and gradients
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

        # Store checkpoint
        ckpt_state = {'epoch': epoch + 1, 'state_dict': model.state_dict()}
        save_checkpoint(ckpt_state, ckpt_file)

        # Compute validation accuracy
        model.eval()
        test_iter = tqdm(testloader, desc='Test @ Epoch %03d' % epoch, leave=False)
        val_accuracy = 0.
        val_loss = 0.
        val_samples = 0
        for inputs, labels in test_iter:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            val_loss += loss_function(outputs, labels).item()
            __, argmax = torch.max(outputs, 1)
            val_accuracy += (labels == argmax.squeeze()).float().sum()
            val_samples += outputs.shape[0]

        val_loss = val_loss / val_samples
        val_accuracy = val_accuracy / val_samples
        # Log validation loss and accuracy
        logger.scalar_summary('validation/loss', val_loss, epoch + 1)
        logger.scalar_summary('validation/accuracy', val_accuracy, epoch + 1)

    return model


def get_accuracy(outputs, labels):
    __, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()
    return accuracy


def test(model, testloader, loss_function, device):
    r"""Computes the accuracy and loss of the model for a given datatset.

    Arguments:
        model (:class:`torch.nn.Module`): nn model
        lossf
        testloader (:class:`torch.utils.data.DataLoader`): dataloader to
            iterate through the data
        loss_function (:class:`torch.nn._Loss`): function to compute loss
        device (:class:`torch.device`): the device to run inference on

    Returns:
        accuracy (float): accuracy of the network on given dataset
    """
    model.eval()
    model.to(device)

    engine = Engine()

    def compute_loss(data):
        """Computes the loss from a given nn model."""
        inputs = data[0]
        labels = data[1]
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        return loss_function(outputs, labels), outputs

    def on_start(state):
        print("Running inference ...")
        state['iterator'] = tqdm(state['iterator'], leave=False)

    class Accuracy():
        _accuracy = 0.;
        _sample_size = 0.

    def on_forward(state):
        batch_size = state['sample'][1].shape[0]
        Accuracy._sample_size += batch_size
        Accuracy._accuracy += batch_size * get_accuracy(state['output'].cpu(), state['sample'][1].cpu())

    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward

    engine.test(compute_loss, testloader)

    return Accuracy._accuracy / Accuracy._sample_size


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(model, filename):
    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint {} trained for {} epochs.'.format(filename, epoch))
    return epoch, model


class AvgClassAugment():

    def __init__(self, dataloader, num_classes, epoch_limit, device):
        self.device = device
        self.dataloader = dataloader
        self.data_iter = None
        self.class_data = {}

        self.batch_num = 0
        self.epoch_limit = epoch_limit

        self.batches_per_epoch = dataloader.__len__()

        # Store a single image from each class in class_data dict
        found_classes = 0
        for data, labels in dataloader:
            if found_classes == num_classes:
                break

            batch_size = labels.shape[0]
            for i in range(batch_size):
                class_id = labels[i].item()
                if not class_id in self.class_data:
                    self.class_data[class_id] = data[i].clone().to(device)
                    found_classes += 1
                    if found_classes == num_classes:
                        break

        data_shape = iter(dataloader).next()[0][0].shape
        self.temp_storage = torch.zeros(data_shape).to(device)

    def __len__(self):
        return self.batches_per_epoch

    def reset(self):
        self.batch_num = 0
        return self

    def __iter__(self):
        self.data_iter = iter(self.dataloader)
        return self

    def __next__(self):
        sample = self.data_iter.next()
        epoch = int(self.batch_num / self.batches_per_epoch)

        data = sample[0].to(self.device)
        labels = sample[1].to(self.device)

        for i in range(labels.shape[0]):
            if epoch < random.randint(1, self.epoch_limit):
                class_id = labels[i].item()

                # Update class_id image in self.class_data
                self.temp_storage.copy_(self.class_data[class_id])
                self.class_data[class_id].copy_(data[i])
                # Average two images
                data[i].add_(self.temp_storage).mul_(0.5)

        self.batch_num += 1
        return [data, labels]


def fuse_conv_bnorm(conv, bn):
    with torch.no_grad():
        # init
        # fusedconv = torch.nn.Conv2d(
        conv_class = type(conv)
        fusedconv = conv_class(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True,
            groups=conv.groups
        )

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(b_conv + b_bn)

        return fusedconv