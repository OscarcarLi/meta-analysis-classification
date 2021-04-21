import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli


class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        # (block_size, block_size) size blocks to be zeroed out
        self.block_size = block_size
        #self.gamma = gamma
        #self.bernouli = Bernoulli(gamma)

    def forward(self, x, gamma):
        """give a 4-d tensor, apply dropblock

        Args:
            x (torch.Tensor): (batch_size, num_channel, h, w)
            gamma (float): the probability of each upper left corner of a block to be zeroed out
                            a rough estimate of how to set this value is given in the dropblock paper

        Returns:
            torch.Tensor: x with each channel's (block_size, block_size) blocks randomly zeroed out.
        """
        if self.training:
            batch_size, channels, height, width = x.shape
            
            bernoulli = Bernoulli(gamma)
            # mask is indicators of the upper left corner of the blocks to be zeroed out
            mask = bernoulli.sample(sample_shape=(batch_size,
                                                  channels,
                                                  height - (self.block_size - 1),
                                                  width - (self.block_size - 1))).to(x.device)
            #print((x.sample[-2], x.sample[-1]))
            block_mask = self._compute_block_mask(mask)
            #print (block_mask.size())
            #print (x.size())
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)
        
        batch_size, channels, height, width = mask.shape
        # the mask has removed some space on each space to avoid boundary issues
        # so we pad it back for the final output
        padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        # returns a list of [(b_idx, c_idx, h_idx, w_idx)...] of all the nonzero locations in mask
        non_zero_idxs = torch.nonzero(mask)
        # count the number of nonzero locations
        nr_blocks = non_zero_idxs.shape[0]

        # offsets is a (2, block_size^2) tensor whose each row is the nonnegative
        # offset to be added to the upper left corner of the zeroing block
        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), # - left_padding,
                torch.arange(self.block_size).repeat(self.block_size), #- left_padding
            ]
        ).t().to(mask.device)
        # the batch and channel dimension will not require additional zeroing, the offsets are for zeroing out additional
        # locations in the height and width dimension
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).to(mask.device).long(), offsets.long()), 1)
        
        if nr_blocks > 0:
            # both of these two location tensors are now of the shape:
            #   (nr_blocks * block_size^2, 4)
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)

            block_idxs = non_zero_idxs + offsets
            
            # zero pad on the last two dimension (h, w)
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            # set the indicator for the additional locations to be zeroed out later
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
            
        block_mask = 1 - padded_mask
        return block_mask