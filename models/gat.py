import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class GAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):

        print('input: ', inputs.shape)
        # print('nb_classes: ', nb_classes)
        # print('attn_drop: ', attn_drop)
        # print('ffd_drop: ', ffd_drop)
        # print('bias_mat: ', bias_mat)
        # print('hid_units: ', hid_units)
        # print('n_heads: ', n_heads)

        # input.shape: (1, 2708, 1433)
        # nb_classes: 7, bias_mat.shape: (1, 2708, 2708)
        # hid_units: [8], n_heads: [8, 1]
        attns = []
        # n_heads[0] : 8
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                # (B,N,D) -> (B,N,F)
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            # (B,N,F1*H1), F1: feature number of node, H1: head number
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                out_sz=nb_classes, activation=lambda x: x,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits
