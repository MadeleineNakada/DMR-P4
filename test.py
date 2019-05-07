import pygame
import pygame.font
pygame.init()

# Colours
BLACK   = (  0,  0,  0)
WHITE   = (255,255,255)
GREEN   = (  0,255,  0)
RED     = (255,  0,  0)
BLUE    = (  0,  0,255)

# Dimensions of screen
size = (400,500)
WIDTH = 500
HEIGHT = 400
screen = pygame.display.set_mode(size)

# Loop Switch
done = False

# Screen Update Speed (FPS)
clock = pygame.time.Clock()

# ------- Main Program Loop -------
while not done:
    # --- Main Event Loop ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    screen.fill(GREEN)
    pygame.draw.rect(screen,(78,203,245),(0,0,250,500),5)


    pygame.display.flip()
    pygame.display.update()

    #Setting FPS
    clock.tick(60)

#Shutdown
pygame.quit()