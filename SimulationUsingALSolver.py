import pygame as pg
import numpy as np
import AugmentedLagrangianConstraintSimulator as als
def toRGB(arr, cold_color = (70,130,200), hot_color = (255,0,0)):
    res = np.empty((arr.shape[0], arr.shape[1], 3), dtype = np.uint8)
    res[:, :, 0] = np.uint8( cold_color[0] * (1 - arr) + hot_color[0] * arr)
    res[:, :, 1] = np.uint8( cold_color[1] * (1 - arr) + hot_color[1] * arr)
    res[:, :, 2] = np.uint8( cold_color[2] * (1 - arr) + hot_color[2] * arr)
    return res

"""Initiating pygame"""
pg.init()
clock = pg.time.Clock()

"""Setting the window size and other parameters"""
scr_size = np.array((1000, 1000), dtype = int)
scr = pg.display.set_mode(scr_size)
pg.display.set_caption("Heat Transfert Simulation")
pg.display.set_icon(pg.surface.Surface((1, 1)))

"""Setting the cold disk parameters"""
circlepos = (0.13, 0.5)
circleradius = 0.1

"""setting the solver grid size by dividing the current window size"""
solver_screen_resolution_ratio = 10
simulation = als.Simulator(int(scr_size[0]/solver_screen_resolution_ratio), int(scr_size[1]/solver_screen_resolution_ratio), circlepos= circlepos, circleradius= circleradius)
print("solver grid size : ", scr_size/solver_screen_resolution_ratio)

"""Setting the time steps"""
dt = 0.00001
NT = 10

"""Starting simulation"""
running = True
t = 0
while running:
    """managing the pygame events so we can stop the simulation by closing the window"""
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
            print("Simulation Manually Stopped")

    """Solving several time steps"""
    for i in range(NT):
        simulation.tick(dt)
        t+=dt

    """Displaying the current solution"""
    scr.blit(pg.transform.smoothscale(pg.surfarray.make_surface(toRGB(simulation.getU().T)), scr_size), (0, 0))

    """Drawing the black cold circle"""
    pg.draw.circle(scr, (0, 0, 0), (scr_size[0]*0.13, scr_size[1]*0.5), scr_size[0]*0.1)

    """Updating the window"""
    pg.display.set_caption("Heat Transfert Simulation | t = "+str(np.round(t, 3)))
    pg.display.flip()

    """Waiting before restarting so we get 60 fps"""
    clock.tick(60)