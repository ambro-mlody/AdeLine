import random
import sys
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


def rand_clicked(self):
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


def move_left(self):
    grid = to_2d()
    grid, col = grid[:, 1:], grid[:, 0]
    grid = np.c_[grid, col]
    color_canvas(grid)


def move_right(self):
    grid = to_2d()
    grid, col = grid[:, :-1], grid[:, -1]
    grid = np.c_[col, grid]
    color_canvas(grid)


def move_up(self):
    grid = to_2d()
    grid, col = grid[1:, :], grid[0, :]
    grid = np.r_[grid, [col]]
    color_canvas(grid)


def move_down(self):
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


def clean_clicked(self):
    for pixel in canvas:
        if pixel.clicked == 1:
            pixel.bg_color = "white"
            pixel.clicked = 0


menu.append(button.Button(position, button_size, "clean", font, color="black", bg_color="gray", on_click=clean_clicked))


def check_clicked(self):
    x = []
    for pixel in canvas:
        x.append(pixel.clicked)

    x = np.concatenate([x, adeline.fourier(x)])
    _max = (0, 0)
    for i in range(10):
        print(perceptrons[i].output(x))
        confidence = perceptrons[i].output(x)
        if confidence > _max[1]:
            _max = (i, confidence)

    output_screen.change_text(str(_max[0]))


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


def learn_clicked(self):
    for i in range(10):
        perceptrons[i].train(data.data, data.labels[i])

    print("done")
    learning_plot()


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
    perceptrons.append(adeline.Adeline(canv_size, sigm=True))

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

    for p in canvas:
        p.draw(window)

    for b in menu:
        b.draw(window)
    output_screen.draw(window)
    py.display.update()
