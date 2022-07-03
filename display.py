#!/usr/bin/env python

import sdl2.ext

class Display(object):
    def __init__(self,W,H):
        sdl2.ext.init()
        self.W , self.H = W, H
        self.window = sdl2.ext.Window("SaqSlam", size=(W,H))
        self.window.show()
    
    def show(self, img):
        events = sdl2.ext.get_events()
        for e in events:
            if e.type == sdl2.SDL_QUIT:
                exit(0)
        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:,:,0:3] = img.swapaxes(0,1)
        self.window.refresh()