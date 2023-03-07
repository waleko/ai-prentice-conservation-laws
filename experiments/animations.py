import numpy as np
from PIL import Image, ImageDraw

width = 512
center = width // 2
color_1 = (0, 0, 0)
color_2 = (255, 255, 255)


def get_tension_color(perc):
    return min(255, int(255 * (1 + perc))), int(255 * (1 - abs(perc))), min(255, int(255 * (1 - perc)))


def pendulum_animator(traj: np.ndarray):
    images = []
    step = 5

    joint = (center, width // 3)
    L = width // 3
    r = width // 10

    for i in range(0, 1000, step):
        alpha, _ = traj[i]
        im = Image.new('RGB', (width, width), color_1)
        draw = ImageDraw.Draw(im)
        ball_center = (joint[0] + L * np.sin(alpha), joint[1] + L * np.cos(alpha))
        draw.line((joint, ball_center), width=1)
        draw.ellipse(((ball_center[0] - r, ball_center[1] - r), (ball_center[0] + r, ball_center[1] + r)), width=1)
        images.append(im)
    return images


def harmonic_oscillator_animator(traj: np.ndarray):
    images = []
    step = 5

    joint = (center, width // 6)
    L = width // 3
    a = width // 20

    maxX = np.max(traj[:, 0])

    for i in range(0, 1000, step):
        raw_x, _ = traj[i]
        x = raw_x * (width // 4) // maxX

        im = Image.new('RGB', (width, width), color_1)
        draw = ImageDraw.Draw(im)
        box_center = (center, joint[1] + x + L)
        draw.line((joint, box_center), fill=get_tension_color(raw_x / maxX))
        draw.rectangle(((box_center[0] - a, box_center[1] - a), (box_center[0] + a, box_center[1] + a)))
        images.append(im)
    return images


def double_pendulum_animator(traj: np.ndarray):
    images = []
    step = 1

    joint = (center, width // 10)
    L = width // 3
    r = width // 20

    for i in range(0, 1000, step):
        theta1, theta2, _, _ = traj[i]
        im = Image.new('RGB', (width, width), color_1)
        draw = ImageDraw.Draw(im)
        split_center = (joint[0] + L * np.sin(theta1), joint[1] + L * np.cos(theta1))
        ball_center = (split_center[0] + L * np.sin(theta2), split_center[1] + L * np.cos(theta2))
        draw.line((joint, split_center))
        draw.line((split_center, ball_center))
        draw.ellipse(((split_center[0] - r, split_center[1] - r), (split_center[0] + r, split_center[1] + r)))
        draw.ellipse(((ball_center[0] - r, ball_center[1] - r), (ball_center[0] + r, ball_center[1] + r)))
        images.append(im)
    return images


def coupled_oscillator_animator(traj: np.ndarray):
    images = []
    step = 1

    a = width // 10
    quarter = width // 4

    xMax = np.max(traj[:, 0])

    for i in range(0, 1000, step):
        raw_x1, raw_x2, _, _ = traj[i]
        x1 = width // 4 + raw_x1 * (width // 4 - a) // xMax
        x2 = width * 3 // 4 + raw_x2 * (width // 4 - a) // xMax

        im = Image.new('RGB', (width, width), color_1)
        draw = ImageDraw.Draw(im)
        draw.line(((0, center + a), (width, center + a)))
        b1_center = (x1, center)
        b2_center = (x2, center)
        draw.line(((0, center), (x1 - a, center)), fill=get_tension_color((x1 - a) / (quarter - a) - 1))
        draw.line(((x1 + a, center), (x2 - a, center)),
                  fill=get_tension_color((x2 - x1 - a * 2) / (center - a * 2) - 1))
        draw.line(((x2 + a, center), (width, center)), fill=get_tension_color((width - x2 - a) / (quarter - a) - 1))
        draw.rectangle(((b1_center[0] - a, b1_center[1] - a), (b1_center[0] + a, b1_center[1] + a)))
        draw.rectangle(((b2_center[0] - a, b2_center[1] - a), (b2_center[0] + a, b2_center[1] + a)))
        images.append(im)
    return images


def kepler_problem_animator(traj: np.ndarray):
    images = []
    step = 8

    sun = (center, center)
    R = width // 50
    r = width // 50
    a = width // 10

    maxX = np.max(np.abs(traj[:, 0]))
    maxY = np.max(np.abs(traj[:, 1]))
    maxD = max(maxX, maxY)

    for i in range(0, 1000, step):
        raw_x, raw_y, _, _ = traj[i]
        x = center + raw_x * (width // 2 - a) // maxD
        y = center + raw_y * (width // 2 - a) // maxD

        im = Image.new('RGB', (width, width), color_1)
        draw = ImageDraw.Draw(im)
        draw.ellipse(((sun[0] - R, sun[1] - R), (sun[0] + R, sun[1] + R)), width=3)
        draw.ellipse(((x - r, y - r), (x + r, y + r)))
        images.append(im)
    return images
