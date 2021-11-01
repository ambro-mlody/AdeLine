import pygame as py


class Button(object):

    def __init__(self, position, size, text, font_size=12,
                 color="white", bg_color="black", on_click=lambda self: print("not implemented")):
        self.on_click = on_click
        self.position = position
        self.size = size
        self.text = text
        self.color = color
        self.bg_color = bg_color
        self.font = py.font.SysFont("Arial", font_size)
        self.text_render = self.font.render(text, 1, color)
        x, y, w, h = self.text_render.get_rect()
        self.rect = py.Rect(position, size)
        self.text_pos = (position[0] + (size[0] - w) / 2, position[1] + (size[1] - h) / 2)
        self.clicked = 0

    def change_text(self, text):
        self.text = text
        self.text_render = self.font.render(text, 1, self.color)
        x, y, w, h = self.text_render.get_rect()
        self.rect = py.Rect(self.position, self.size)
        self.text_pos = (self.position[0] + (self.size[0] - w) / 2, self.position[1] + (self.size[1] - h) / 2)

    def draw(self, window):
        py.draw.rect(window, self.bg_color, self.rect)
        window.blit(self.text_render, self.text_pos)

    def click(self):
        if self.rect.collidepoint(py.mouse.get_pos()):
            self.on_click(self)
