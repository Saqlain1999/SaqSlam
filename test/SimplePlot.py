import pygame

window = pygame.display.set_mode((1024, 768))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
    window.fill((0, 0, 0))
    pygame.display.update()
    pygame.time.delay(10)