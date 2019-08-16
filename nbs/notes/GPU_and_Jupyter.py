#!/usr/bin/env python
# coding: utf-8

# # Trying out the GPU



import torch




torch.cuda.is_available()




t_cpu = torch.rand(100, 100, 100)




get_ipython().run_line_magic('timeit', 't_cpu @ t_cpu')




t_gpu = torch.rand(100, 100, 100).cuda()  # OOM for 500x500x500?




get_ipython().run_line_magic('timeit', 't_gpu @ t_gpu')


# # Jupyter notebook tips


# - [] Task list
#     - [x] Done!



get_ipython().run_line_magic('pinfo', 'Function')




get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# works by reloading all modules before running a cell



get_ipython().run_line_magic('debug', '')







