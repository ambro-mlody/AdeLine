import random
import sys

import matplotlib.colors
import pygame as py
import button
import data
import adeline
import numpy as np
import matplotlib.pyplot as plt

py.init()
size = (490, 670)
window = py.display.set_mode(size)
py.display.set_caption("AdeLine")


def number_clicked(self):
    d = data.data[int(self.text)]
    i = 0
    for p in canvas:
        p.clicked = d[i]
        if d[i] == 0:
            p.bg_color = "white"
        else:
            p.bg_color = "red"
        i += 1


canv_size = 63
font = 34
button_size = (40, 40)
menu = []
x = 6
y = 581
diff = 43
position = (x, y)
for i in range(5):
    menu.append(button.Button(position, button_size, str(i), font, bg_color="blue", on_click=number_clicked))
    position = (position[0] + diff, position[1])

position = (x, y + diff)
for i in range(5, 10):
    menu.append(button.Button(position, button_size, str(i), font, bg_color="blue", on_click=number_clicked))
    position = (position[0] + diff, position[1])


def rand_clicked(self=None):
    for pixel in canvas:
        prob = 0.025
        rand = random.random()
        if rand < prob:
            if pixel.clicked == 0:
                pixel.clicked = 1
                pixel.bg_color = "red"
            else:
                pixel.clicked = 0
                pixel.bg_color = "white"


def to_2d():
    array = []
    for p in canvas:
        array.append(p.clicked)

    grid = np.array(array)
    grid = grid.reshape((9, 7))
    return grid


def color_canvas(grid):
    grid = grid.reshape(canv_size)
    for i in range(0, canv_size):
        canvas[i].clicked = grid[i]
        if grid[i] == 0:
            canvas[i].bg_color = "white"
        else:
            canvas[i].bg_color = "red"


def move_left(self=None):
    grid = to_2d()
    grid, col = grid[:, 1:], grid[:, 0]
    grid = np.c_[grid, col]
    color_canvas(grid)


def move_right(self=None):
    grid = to_2d()
    grid, col = grid[:, :-1], grid[:, -1]
    grid = np.c_[col, grid]
    color_canvas(grid)


def move_up(self=None):
    grid = to_2d()
    grid, col = grid[1:, :], grid[0, :]
    grid = np.r_[grid, [col]]
    color_canvas(grid)


def move_down(self=None):
    grid = to_2d()
    grid, col = grid[:-1, :], grid[-1, :]
    grid = np.r_[[col], grid]
    color_canvas(grid)


font = 20
button_size = (50, 40)
diff_x = 53

menu.append(button.Button(position, button_size, "←", font, color="black", bg_color="yellow", on_click=move_left))
position = (position[0] + diff_x, position[1])

menu.append(button.Button(position, button_size, "↓", font, color="black", bg_color="yellow", on_click=move_down))
position = (position[0] + diff_x, position[1])

menu.append(button.Button(position, button_size, "→", font, color="black", bg_color="yellow", on_click=move_right))
position = (position[0] - 2 * diff_x, position[1] - diff)

menu.append(button.Button(position, button_size, "rand", font, color="black", bg_color="gray", on_click=rand_clicked))
position = (position[0] + diff_x, position[1])

menu.append(button.Button(position, button_size, "↑", font, color="black", bg_color="yellow", on_click=move_up))
position = (position[0] + diff_x, position[1])


def clean_clicked(self=None):
    for pixel in canvas:
        if pixel.clicked == 1:
            pixel.bg_color = "white"
            pixel.clicked = 0


menu.append(button.Button(position, button_size, "clean", font, color="black", bg_color="gray", on_click=clean_clicked))
confidence = []


def predict():
    x = []
    for pixel in canvas:
        x.append(pixel.clicked)

    confidence.clear()
    x = np.concatenate([x, adeline.fourier(x)])
    _max = (0, 0)
    for i in range(10):
        # print(perceptrons[i].output(x))
        out = perceptrons[i].output(x)
        confidence.append(out)
        if out > _max[1]:
            _max = (i, out)
    return _max[0]


def check_clicked(self=None):
    output_screen.change_text(str(predict()))


position = (position[0] + diff_x, position[1] + diff)
menu.append(button.Button(position, button_size, "check", font, color="black", bg_color="gray", on_click=check_clicked))
position = (position[0], position[1] - diff)


def learning_plot():
    plt.ylabel("Error Value")
    plt.xlabel("iteration")
    for i in range(10):
        plt.plot(perceptrons[i].errors[1:100])
    plt.legend([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.show()


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def confidence_plot():
    nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    temp = np.array(confidence)
    temp = normalize(temp)
    plt.bar(nums, temp)
    plt.show()


def learn_clicked(self=None):
    for i in range(10):
        perceptrons[i].train(data.data, data.labels[i])
    print("done")


def matrix_plot():
    matrix = np.zeros((10, 10))
    for i in range(10):
        num_key(i)
        matrix[i, predict()] += 1
        move_up()
        matrix[i, predict()] += 1
        move_left()
        matrix[i, predict()] += 1
        move_down()
        matrix[i, predict()] += 1
        move_down()
        matrix[i, predict()] += 1
        move_right()
        matrix[i, predict()] += 1
        move_right()
        matrix[i, predict()] += 1
        move_up()
        matrix[i, predict()] += 1
        move_up()
        matrix[i, predict()] += 1

    matrix /= 9
    plt.imshow(matrix, cmap="magma")
    plt.ylabel("Label")
    plt.xlabel("Prediction")
    plt.colorbar()
    plt.show()


menu.append(button.Button(position, button_size, "learn", font, color="black", bg_color="gray", on_click=learn_clicked))

position = (position[0] + diff_x, position[1])
button_size = (size[0] - position[0] - x, size[1] - position[1] - x)
font = 60
output_screen = button.Button(position, button_size, "", font, color="black", bg_color="green")


def pixel_clicked(self):
    if self.clicked == 0:
        self.bg_color = "red"
        self.clicked = 1
    else:
        self.bg_color = "white"
        self.clicked = 0


button_size = (60, 60)
x = 30
position = (x, 15)
diff = 62
canvas = []
for i in range(1, canv_size + 1):
    canvas.append(button.Button(position, button_size, "", font, bg_color="white", on_click=pixel_clicked))
    position = (position[0] + diff, position[1])
    if i % 7 == 0:
        position = (x, position[1] + diff)

perceptrons = []
for i in range(10):
    perceptrons.append(adeline.Adeline(canv_size))


def num_key(num):
    d = data.data[num]
    i = 0
    for p in canvas:
        p.clicked = d[i]
        if d[i] == 0:
            p.bg_color = "white"
        else:
            p.bg_color = "red"
        i += 1


def switch(key):
    if key == py.K_0:
        num_key(0)
    elif key == py.K_1:
        num_key(1)
    elif key == py.K_2:
        num_key(2)
    elif key == py.K_3:
        num_key(3)
    elif key == py.K_4:
        num_key(4)
    elif key == py.K_5:
        num_key(5)
    elif key == py.K_6:
        num_key(6)
    elif key == py.K_7:
        num_key(7)
    elif key == py.K_8:
        num_key(8)
    elif key == py.K_9:
        num_key(9)
    elif key == py.K_l:
        learn_clicked()
    elif key == py.K_c:
        clean_clicked()
    elif key == py.K_r:
        rand_clicked()
    elif key == py.K_RETURN:
        check_clicked()
    elif key == py.K_LEFT:
        move_left()
    elif key == py.K_RIGHT:
        move_right()
    elif key == py.K_UP:
        move_up()
    elif key == py.K_DOWN:
        move_down()
    elif key == py.K_p:
        learning_plot()
    elif key == py.K_s:
        confidence_plot()
    elif key == py.K_m:
        matrix_plot()


while True:
    window.fill((0, 0, 0))
    for ev in py.event.get():
        if ev.type == py.QUIT:
            py.quit()
            sys.exit(0)
        if py.mouse.get_pressed()[0]:
            for b in menu:
                b.click()
            for p in canvas:
                p.click()
        if ev.type == py.KEYDOWN:
            switch(ev.key)

    for p in canvas:
        p.draw(window)

    for b in menu:
        b.draw(window)
    output_screen.draw(window)
    py.display.update()
