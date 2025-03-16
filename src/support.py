import os
import numpy as np
import pandas as pd
import seaborn as sns
import math

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import *
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as keras
from keras_core import callbacks
from keras import ops
from keras import layers
from keras import optimizers
import keras_tuner
from tensorflow_probability import distributions as tfd
from scipy.stats import linregress

from scipy.special import erf, erfinv
from matplotlib.colors import Normalize
from matplotlib import colors
from matplotlib import cm
import itertools
from scipy.stats import entropy
from scipy.interpolate import interpn

kB = 1.38064852e-23
conv_v = 1.0e2


def compute_AC_correlation_method(vel: np.array,gas_type: str,direction: str):
    """Function that computes Different ACs using correlation method for monoatomic and diatomic gases"""
    mono=['Ar','He']
    MAC_x=1-np.polyfit(vel[:,0],vel[:,3],1)[0]
    EAC_x=1-np.polyfit((vel[:,0])*2,(vel[:,3])*2,1)[0]
    EAC_y=1-np.polyfit((vel[:,1])*2,(vel[:,4])*2,1)[0]
    EAC_z=1-np.polyfit((vel[:,2])*2,(vel[:,5])*2,1)[0]
    EAC_tr=1-np.polyfit(((vel[:,0])*2+(vel[:,1])*2+(vel[:,2])*2),((vel[:,3])*2+(vel[:,4])*2+(vel[:,5])*2),1)[0]
    if gas_type in mono:
        if direction == 'y':
            MAC_y = 1 - np.polyfit(np.abs(vel[:, 1]), np.abs(vel[:, 4]), 1)[0]
            MAC_z = 1 - np.polyfit(vel[:, 2], vel[:, 5], 1)[0]
        if direction=='z':
            MAC_y = 1 - np.polyfit(vel[:, 1], vel[:, 4], 1)[0]
            MAC_z = 1 - np.polyfit(np.abs(vel[:, 2]), np.abs(vel[:, 5]), 1)[0]
        return MAC_x,MAC_y,MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr

    if gas_type=='H2':
        mg=2*1.0079*0.001/(6.022e23) #[kg]
        mu=mg/4
        b_l=0.741e-10 #[m]
        I=mu*(b_l**2)
        conv_omega=1.0e12 #convert [1/ps] to [1/s]
        conv_J_2_eV=6.24e18
        vel_tr_SI=vel[:,0:6]*conv_v
        omega_SI=vel[:,6:10]*conv_omega
        tr_energy_in=0.5*mg*(np.linalg.norm(vel_tr_SI[:,0:3],axis=1))**2
        tr_energy_out=0.5*mg*(np.linalg.norm(vel_tr_SI[:,3:6],axis=1))**2
        rot_energy_in=0.5*I*(omega_SI[:,0]*2+omega_SI[:,1]*2)
        rot_energy_out=0.5*I*(omega_SI[:,2]*2+omega_SI[:,-1]*2)
        tot_energy_in=tr_energy_in+rot_energy_in
        tot_energy_out=tr_energy_out+rot_energy_out
        EAC_tr_energy=1-np.polyfit(tr_energy_in[:]*conv_J_2_eV,tr_energy_out[:]*conv_J_2_eV,1)[0]
        EAC_rot=1-np.polyfit(((vel[:,6])*2+(vel[:,7])*2),((vel[:,8])*2+(vel[:,9])*2),1)[0]
        EAC_rot_energy=1-np.polyfit(rot_energy_in[:]*conv_J_2_eV,rot_energy_out[:]*conv_J_2_eV,1)[0]
        EAC_tot_energy=1-np.polyfit(tot_energy_in[:]*conv_J_2_eV,tot_energy_out[:]*conv_J_2_eV,1)[0]
        if direction=='y':    
            MAC_y=1-np.polyfit(np.abs(vel[:,1]),np.abs(vel[:,4]),1)[0]
            MAC_z=1-np.polyfit(vel[:,2],vel[:,5],1)[0]    
        if direction=='z':
            MAC_y=1-np.polyfit(np.abs(vel[:,1]),np.abs(vel[:,4]),1)[0]
            MAC_z=1-np.polyfit(vel[:,2],vel[:,5],1)[0]
        return MAC_x,MAC_y,MAC_z,EAC_x,EAC_y,EAC_z,EAC_tr,EAC_tr_energy,EAC_rot,EAC_rot_energy,EAC_tot_energy



# Function to re-normalize data
def re_normalize(x, min, max):
    return (x * (max - min)) + min


# Function to fit least squares line and plot
def calculate_AC_contor(df, cols_pairs):
    num_plots = len(cols_pairs)
    cols = 2
    rows = (num_plots + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten in case of single row

    slopes_intercepts = []

    for idx, (x_col, y_col) in enumerate(cols_pairs):
        x = df[x_col]
        y = df[y_col]

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Calculate AC
        AC = 1 - slope

        # Store slope and intercept
        slopes_intercepts.append((x_col, y_col, slope, intercept, AC))

        # Plot filled contour density plot

        sns.kdeplot(x=x, y=y, fill=True, ax=axes[idx])

        # Plot the regression line
        axes[idx].plot(x, slope * x + intercept, color='red', label=f'Fit Line: y = {slope:.4f}x + {intercept:.4f}')
        axes[idx].set_xlabel(x_col)
        axes[idx].set_ylabel(y_col)
        axes[idx].set_title(f'Filled Contour Plot for {x_col} vs {y_col}')
        axes[idx].legend(loc = 1)
        
        
    # Remove any empty subplots
    for ax in axes[num_plots:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

    # Print slopes, intercepts, and AC
    for x_col, y_col, slope, intercept, AC in slopes_intercepts:
        print(f"Slope, Intercept and AC for {x_col} vs {y_col}: Slope = {slope}, Intercept = {intercept}, AC = {AC}")
    

def calculate_AC_scatter(df, cols_pairs):
    num_plots = len(cols_pairs)
    cols = 2
    rows = (num_plots + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten in case of single row

    slopes_intercepts = []

    for idx, (x_col, y_col) in enumerate(cols_pairs):
        x = df[x_col]
        y = df[y_col]

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Calculate AC
        AC = 1 - slope

        # Store slope and intercept
        slopes_intercepts.append((x_col, y_col, slope, intercept, AC))

        # Plot scatter and line
        axes[idx].scatter(x, y, label='Data Points')
        axes[idx].plot(x, slope * x + intercept, color='red', label=f'Fit Line: y = {slope:.4f}x + {intercept:.4f}')
        axes[idx].set_xlabel(x_col)
        axes[idx].set_ylabel(y_col)
        axes[idx].set_title(f'Least Squares Fit for {x_col} vs {y_col}')
        axes[idx].legend(loc = 1)
    
    # Remove any empty subplots
    for ax in axes[num_plots:]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()

    # Print slopes and intercepts
    for x_col, y_col, slope, intercept, AC in slopes_intercepts:
        print(f"Slope, Intercept and AC for {x_col} vs {y_col}: Slope = {slope}, Intercept = {intercept}, AC = {AC}")


def elu_plus_one_plus_epsilon(x):
    return keras.activations.elu(x) + 1 + keras.backend.epsilon()

class MixtureDensityOutput(layers.Layer):
    def __init__(self, output_dimension, num_mixtures, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dimension
        self.num_mix = num_mixtures
        self.mdn_mus = layers.Dense(
            self.num_mix * self.output_dim, name="mdn_mus"
        )  # mix*output vals, no activation
        self.mdn_sigmas = layers.Dense(
            self.num_mix * self.output_dim,
            activation=elu_plus_one_plus_epsilon,
            name="mdn_sigmas",
        )  # mix*output vals exp activation
        self.mdn_pi = layers.Dense(self.num_mix, name="mdn_pi")  # mix vals, logits

    def build(self, input_shape):
        self.mdn_mus.build(input_shape)
        self.mdn_sigmas.build(input_shape)
        self.mdn_pi.build(input_shape)
        super().build(input_shape)

    @property
    def testable_weights(self):
        return (
            self.mdn_mus.testable_weights
            + self.mdn_sigmas.trainable_weights
            + self.mdn_pi.trainable_weights
        )

    @property
    def non_trainable_weights(self):
        return (
            self.mdn_mus.non_trainable_weights
            + self.mdn_sigmas.non_trainable_weights
            + self.mdn_pi.non_trainable_weights
        )

    def call(self, x, mask=None):
        return layers.concatenate(
            [self.mdn_mus(x), self.mdn_sigmas(x), self.mdn_pi(x)], name="mdn_outputs"
        )
    
def get_mixture_loss_func(output_dim, num_Mixes):
    def mdn_loss_func(y_true, y_pred):
        # Reshape inputs in case this is used in a TimeDistributed layer
        y_pred = tf.reshape(
            y_pred,
            [-1, (2 * num_Mixes * output_dim) + num_Mixes],
            name="reshape_ypreds",
        )
        y_true = tf.reshape(y_true, [-1, output_dim], name="reshape_ytrue")
        # Split the inputs into parameters
        out_mu, out_sigma, out_pi = tf.split(
            y_pred,
            num_or_size_splits=[
                num_Mixes * output_dim,
                num_Mixes * output_dim,
                num_Mixes,
            ],
            axis=-1,
            name="mdn_coef_split",
        )
        # Construct the mixture models
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_Mixes
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [
            tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            for loc, scale in zip(mus, sigs)
        ]
        mixture = tfd.Mixture(cat=cat, components=coll)
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        loss = tf.reduce_mean(loss)
        return loss

    return mdn_loss_func

def split_mixture_params(params, output_dim, num_Mixes):
    mus = params[: num_Mixes * output_dim]
    sigs = params[num_Mixes * output_dim : 2 * num_Mixes * output_dim]
    pi_logits = params[-num_Mixes:]
    return mus, sigs, pi_logits


def softmax(w, t=1.0):
    e = np.array(w) / t  # adjust temperature
    e -= e.max()  # subtract max to protect from exploding exp values.
    e = np.exp(e)
    dist = e / np.sum(e)
    return dist


def sample_from_categorical(dist):
    r = np.random.rand(1)  # uniform random number in [0,1]
    accumulate = 0
    for i in range(0, dist.size):
        accumulate += dist[i]
        if accumulate >= r:
            return i
    tf.logging.info("Error sampling categorical model.")
    return -1


def sample_from_output(params, output_dim, num_Mixes, temp=1.0, sigma_temp=1.0):
    mus, sigs, pi_logits = split_mixture_params(params, output_dim, num_Mixes)
    pis = softmax(pi_logits, t=temp)
    m = sample_from_categorical(pis)
    # Alternative way to sample from categorical:
    # m = np.random.choice(range(len(pis)), p=pis)
    mus_vector = mus[m * output_dim : (m + 1) * output_dim]
    sig_vector = sigs[m * output_dim : (m + 1) * output_dim]
    scale_matrix = np.identity(output_dim) * sig_vector  # scale matrix from diag
    cov_matrix = np.matmul(scale_matrix, scale_matrix.T)  # cov is scale squared.
    cov_matrix = cov_matrix * sigma_temp  # adjust for sigma temperature
    sample = np.random.multivariate_normal(mus_vector, cov_matrix, 1)
    return sample
