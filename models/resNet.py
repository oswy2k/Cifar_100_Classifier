import torch.nn as nn
import torch.nn.functional as F


# Main convolutional block  #
def convolutional_block(in_channels:int, out_channels:int, pool:bool=False)->nn.Sequential:
    """
    Parameters
    ----------
    in_channels : int 
    out_channels : int 
    pool : bool = False, optional

    Returns
    ----------
    nn.Sequential

    Notes
    ----------
    ResNet basic convolutional building block.
    """
    
    ## Basic Convolutionaal Block ##
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    
    ## Max pooling if needed ##
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
    

class ResNet9(nn.Module):
    """
    Parameters
    ----------
    in_channels : int 
    num_classes : int 

    Returns
    ----------
    ResNet9 class Object.

    Notes
    ----------
    Returns an Neural nerwork object, based on the Resnet paper.
    """

    def __init__(self, in_channels:int, num_classes:int):
        super().__init__()
        
        self.conv1 = convolutional_block(in_channels, 64)
        self.conv2 = convolutional_block(64, 128, pool=True) 
        self.res1 = nn.Sequential(convolutional_block(128, 128), 
                                  convolutional_block(128, 128)) 
        
        self.conv3 = convolutional_block(128, 256, pool=True)
        self.conv4 = convolutional_block(256, 512, pool=True) 

        self.res2 = nn.Sequential(convolutional_block(512, 512), 
                                  convolutional_block(512, 512)) 
        
        self.conv5 = convolutional_block(512, 1028, pool=True) 

        self.res3 = nn.Sequential(convolutional_block(1028, 1028), 
                                  convolutional_block(1028, 1028))  
        
        self.classifier = nn.Sequential(nn.MaxPool2d(2), # 1028 x 1 x 1
                                        nn.Flatten(), # 1028 
                                        nn.Linear(1028, num_classes)) # 1028 -> 100
        
    def forward(self, input_batch):
        conv1_output = self.conv1(input_batch)
        conv2_output = self.conv2(conv1_output)
        rest1_output = self.res1(conv2_output) + conv2_output
        conv3_output = self.conv3(rest1_output)
        conv4_output = self.conv4(conv3_output)
        rest2_output = self.res2(conv4_output) + conv4_output
        conv5_output = self.conv5(rest2_output)
        rest3_output = self.res3(conv5_output) + conv5_output
        classified = self.classifier(rest3_output)
        return classified