import os
import torch


# Setup hardware device
def hardware():
  hardware = None
  if torch.has_mps:
    # mps for running on Apple GPU
    hardware = 'mps'
    torch.set_default_dtype(torch.float32)
  elif torch.cuda.is_available():
    # cuda for running on Nvidia
    hardware = 'cuda'
  else:
    hardware = 'cpu'

  return hardware


def create_subdirectories(parent, subs):
    path_str = parent

    # Get path of the current directory
    current_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Create parent directory if it doesn't exist
    parent_dir = os.path.join(current_directory, parent)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    # Create subdirectories inside parent
    current_path = parent_dir
    for subdirectory in subs:
        path_str += f'/{subdirectory}'
        current_path = os.path.join(current_path, subdirectory)
        if not os.path.exists(current_path):
            os.makedirs(current_path)
    
    return path_str


def start_video(counter):
    start = False
    if counter % 1_000_000 == 0:
        start = True
    return start
