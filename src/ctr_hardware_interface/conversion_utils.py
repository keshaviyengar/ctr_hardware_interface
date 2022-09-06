#!/usr/bin/env python

import numpy as np

# Useful conversion to convert between absolute position, relative position and steps for extension and rotation


def alpha2motor(alphas_deg, deg2steps, motor_dir):
    alphas_dir = np.multiply(motor_dir, alphas_deg)
    return np.multiply(alphas_dir, deg2steps)


def motor2alpha(xyz_rot, steps2deg, motor_dir):
    alphas_deg = np.multiply(steps2deg, xyz_rot)
    alphas_dir = np.multiply(motor_dir, alphas_deg)
    return alphas_dir


def beta2motor(beta, mm2steps):
    return np.multiply(beta, mm2steps)


def motor2beta(xyz_tr, steps2mm):
    return np.multiply(xyz_tr, steps2mm)
