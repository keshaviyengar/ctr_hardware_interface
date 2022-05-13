#!/usr/bin/env python

import numpy as np


def write_read(dev, input_val, num_lines, report):
    dev.write(input_val)
    output = []
    while True:
        temp = dev.read_until('\n')
        output.append(temp)
        if len(output) == num_lines - 1:
            break
    if report:
        print("input: " + input_val)
        print("output: " + ' '.join(item for item in output))
    return output


def motor_position(dev, report):
    output = write_read(dev, "?\n".encode("ascii"), 3, report)
    output = output[0].partition("MPos:")[2]
    output = output.split('|')[0]
    positions = [float(x) for x in output.split(',')]
    return positions


def intersecting_lines(x, y, x0, y0):
    # Check for overlapping carriages
    xy = np.array([x, y])
    xy0 = np.array([x0, y0])
    # Travel vector
    dxy = xy - xy0
    # x(t) = y(t)
    # dxy[0]*t + xy[0] = dxy[1]*t + xy[1]
    t_xy = (xy0[0] - xy0[1]) / (dxy[1] - dxy[0])
    print("xy: " + str(xy))
    print("xy0: " + str(xy0))
    print("t_xy: " + str(t_xy))
    # Contact midway or start at the same location and y > x
    if (1 > t_xy > 0) or (t_xy == 0 and xy[0] < xy[1]):
        xy[0] = dxy[0] * t_xy + xy0[0]
        xy[1] = dxy[1] * t_xy + xy0[1]
    return xy


def safe_move(dev, xyz, vel, motor_range, is_safe, report):
    if is_safe:
        # First check single carriage constraints
        for i in range(3):
            xyz[i] = max(xyz[i], motor_range[i, 0])
            xyz[i] = min(xyz[i], motor_range[i, 1])
        # Check for overlap in carriages
        xyz0 = motor_position(dev, report)
        # xy axes test: intersecting lines
        xyz[0:2] = intersecting_lines(xyz[0], xyz[1], xyz0[0], xyz0[1])
        # yz axes test: intersecting lines
        xyz[1:3] = intersecting_lines(xyz[1], xyz[2], xyz0[1], xyz0[2])
        # xy axes test: intersecting lines
        xyz[0:2] = intersecting_lines(xyz[0], xyz[1], xyz0[0], xyz0[1])

    input_val = 'G1 X' + str(xyz[0]) + ' Y' + str(xyz[1]) + ' Z' + str(xyz[2]) + 'F' + str(vel) + '\n'
    write_read(dev, input_val.encode('ascii'), 2, report)

    return xyz


def stop_move(dev, report):
    # Stop
    write_read(dev, "#\n", 4, report)
    # Unlock
    write_read(dev, "$X\n", 2, report)


def initialize_cnc(dev_rot, dev_tr, report, is_lock=False):
    # Rotational
    print("connected to: " + dev_rot.name)
    print(dev_rot.readlines(2))
    # Unlock rotational motors
    write_read(dev_rot, "$1=25\n".encode("ascii"), 2, report)

    # Translational
    print("connected to: " + dev_tr.name)
    print(dev_tr.readlines(2))

    # Translational homing
    write_read(dev_tr, "$H\n".encode("ascii"), 3, report)
    # Trans position should be x-1 y-1 z-1
    print('translation (-1.0, -1.0, -1.0): ' + str(motor_position(dev_tr, True)))
    # Rot position should be x0 y0 z0
    print('rotation (0.0, 0.0, 0.0): ' + str(motor_position(dev_rot, True)))
    # Set to absolute position g90 (relative is g91)
    write_read(dev_tr, "g90\n".encode("ascii"), 2, report)
    write_read(dev_rot, "g90\n".encode("ascii"), 2, report)

    write_read(dev_tr, "g1 x-2 y-4 z-6 f1000\n".encode("ascii"), 3, report)
    print('translation: ' + str(motor_position(dev_tr, report)))

    if is_lock:
        write_read(dev_rot, '$1=255\n', 2, report)

    return True
