#!/usr/bin/env python

import numpy as np

# Useful conversion to convert between absolute position, relative position and steps for extension and rotation


def alpha2motor(alphas, deg2steps, motor_dir):
    alphas_deg = np.rad2deg(alphas)
    alphas_dir = np.multiply(motor_dir, alphas_deg)
    return np.multiply(alphas_dir, deg2steps)


def motor2alpha(xyz_rot, steps2deg, motor_dir):
    alphas_deg = np.multiply(steps2deg, xyz_rot)
    alphas_dir = np.multiply(motor_dir, alphas_deg)
    return np.deg2rad(alphas_dir)


def rel_beta2motor(rel_beta, mm2steps):
    return np.multiply(rel_beta, mm2steps)


def motor2rel_beta(xyz_tr, steps2mm):
    return np.multiply(xyz_tr, steps2mm)


def abs_beta2rel_beta(abs_beta, carriage_lengths):
    rel_beta = abs_beta - carriage_lengths
    return rel_beta


def rel_beta2abs_beta(rel_beta, carriage_lengths):
    abs_beta = rel_beta + carriage_lengths
    return abs_beta


def abs_beta2motor(abs_beta, carriage_lengths, mm2steps):
    rel_beta = abs_beta2rel_beta(abs_beta, carriage_lengths)
    return rel_beta2motor(rel_beta, mm2steps)


def motor2abs_beta(xyz_tr, carriage_lengths, steps2mm):
    rel_beta = motor2rel_beta(xyz_tr, steps2mm)
    return rel_beta2abs_beta(rel_beta, carriage_lengths)


